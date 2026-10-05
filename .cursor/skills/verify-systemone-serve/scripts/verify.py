"""Launch, check, drive, and clean up `sys1` for verification, keeping evidence that survives cleanup.

Usage and the evidence layout are in ../SKILL.md; `verify.py <command> -h` lists each command's options. The repo is
found from this file's location (override with VERIFY_REPO). Placeholders in `launch`/`cli` arguments and `--env`
values: {port} (a fresh free port per call; in `launch`, the instance's own), {port:NAME} and {url:NAME}.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = Path(os.environ.get("VERIFY_REPO") or HERE.parents[4]).resolve()
SKILL = "verify-systemone-serve"
SCRATCH_ROOT = Path(os.environ.get("TMPDIR", "/tmp")) / SKILL
ARTIFACT_ROOT = Path(os.environ.get("VERIFY_ARTIFACTS") or REPO / ".verify-artifacts" / SKILL)
DROP = ("TYPESAFE_BASE_URL", "TOGETHER_MODEL", "SHIM_TOKEN", "HF_TOKEN", "SYSTEMONE_SERVE_API_KEY")
PROBE_HEADER = "x-verify-probe"  # marks the helper's own readiness/doctor requests
STOP_WAIT = 20.0  # serve's own bound is 5 s grace + 2 s stuck requests (+ launched upstreams)


def fail(message: str, code: int = 1) -> None:
    print(json.dumps({"error": message}), file=sys.stderr)
    sys.exit(code)


def python() -> str:
    exe = REPO / ".venv" / "bin" / "python"
    if not exe.exists():
        fail(f"{exe} is missing: run `uv sync --frozen` in {REPO} first")
    return str(exe)


def run_id() -> str:
    rid = os.environ.get("VERIFY_RUN_ID", "")
    if not re.fullmatch(r"[0-9A-Za-z_-]+", rid):
        fail('VERIFY_RUN_ID is not set: run `eval "$(verify.py init)"` first')
    return rid


def scratch() -> Path:
    return SCRATCH_ROOT / run_id()


def evidence() -> Path:
    return ARTIFACT_ROOT / run_id()


def load_state() -> dict:
    path = scratch() / "state.json"
    if not path.exists():
        fail(f"no scratch state at {path}: this run was cleaned up, or `init` was not run")
    return json.loads(path.read_text())


def save_state(state: dict) -> None:
    (scratch() / "state.json").write_text(json.dumps(state, indent=2))


def free_port(state: dict) -> int:
    for _ in range(50):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in state["ports"]:
            state["ports"].append(port)
            return port
    fail("no free port")
    return 0


def child_env(extra: list[str]) -> dict:
    env = {k: v for k, v in os.environ.items() if not (
        k.endswith("_API_KEY") or k in DROP or (k.startswith("SYSTEMONE_") and k.endswith(("_URL", "_MODEL"))))}
    for item in extra:
        key, _, value = item.partition("=")
        env[key] = value
    return env


def expand(text: str, state: dict, port: int | None) -> str:
    def sub(match: re.Match) -> str:
        what, name = match.group(1), match.group(2)
        if name is None:
            return str(port)
        inst = state["instances"].get(name) or fail(f"no instance {name!r} in this run")
        return str(inst["port"]) if what == "port" else inst["url"]
    return re.sub(r"\{(port|url)(?::([A-Za-z0-9_-]+))?\}", sub, text)


def write_evidence(step: str, record: dict) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+(/[A-Za-z0-9_.-]+)*", step):
        fail(f"step {step!r}: use feature/step with letters, digits, '_', '-', '.'")
    path = evidence() / f"{step}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"step": step, "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), **record}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return path


def checks(exprs: list[str], scope: dict) -> list[dict]:
    results = []
    for expr in exprs:
        try:
            ok = bool(eval(expr, {"__builtins__": {"len": len, "any": any, "all": all, "str": str, "int": int,
                                                   "sorted": sorted, "set": set, "isinstance": isinstance}}, scope))
            results.append({"check": expr, "ok": ok})
        except Exception as error:  # noqa: BLE001 - a check that cannot be evaluated is a failed check
            results.append({"check": expr, "ok": False, "error": f"{type(error).__name__}: {error}"})
    return results


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def command_of(pid: int) -> str:
    out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, check=False)
    return out.stdout.strip()


def ours(inst: dict) -> bool:
    """The recorded pid is still the `sys1 serve` (or tap) this run started (pids get reused)."""
    command = command_of(inst["pid"])
    if inst.get("kind") == "tap":
        return "verify.py _tap" in command and f" {inst['port']} " in f"{command} "
    return "systemone_serve.cli serve" in command and f"--port {inst['port']}" in command


def http_get(url: str, timeout: float = 2.0) -> tuple[int, str]:
    try:
        probe = urllib.request.Request(url, headers={PROBE_HEADER: "1"})  # taps do not record these
        with urllib.request.urlopen(probe, timeout=timeout) as reply:
            return reply.status, reply.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")
    except OSError as error:
        return 0, str(error)


def port_owner(port: int) -> str | None:
    """pid listening on 127.0.0.1:port, via ss (Linux) or lsof (macOS); None when neither tool is available."""
    if shutil.which("ss"):
        out = subprocess.run(["ss", "-Hltnp", f"sport = :{port}"], capture_output=True, text=True, check=False).stdout
        found = re.findall(r"pid=(\d+)", out)
        return found[0] if found else ""
    if shutil.which("lsof"):
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True, check=False)
        return out.stdout.split()[0] if out.stdout.split() else ""
    return None


# ---------------------------------------------------------------- commands

def cmd_init(args: argparse.Namespace) -> None:
    rid = datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    os.environ["VERIFY_RUN_ID"] = rid
    scratch().mkdir(parents=True)
    evidence().mkdir(parents=True)
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True, check=False)
    pkg = subprocess.run([python(), "-c", "import systemone_serve as s; print(s.__version__, s.__file__)"],
                         capture_output=True, text=True, env=child_env([]), check=False).stdout.split()
    meta = {"run_id": rid, "repo": str(REPO), "git_head": git.stdout.strip() or "not a git checkout",
            "git_dirty_files": len(dirty.stdout.splitlines()) if git.returncode == 0 else None,
            "package": pkg, "python": python(), "scratch": str(scratch()), "evidence": str(evidence()),
            "keys_in_agent_env": sorted(k for k in os.environ if k.endswith("_API_KEY")),  # names only, never values
            "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    (evidence() / "run.json").write_text(json.dumps(meta, indent=2))
    save_state({"instances": {}, "ports": [], "order": []})
    print(f"export VERIFY_RUN_ID={rid}")
    print(f"# evidence: {evidence()}")


def cmd_port(args: argparse.Namespace) -> None:
    state = load_state()
    print(free_port(state))
    save_state(state)


def cmd_launch(args: argparse.Namespace) -> None:
    state = load_state()
    name = args.name
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in state["instances"]:
        fail(f"instance name {name!r} is invalid or already used in this run")
    port = args.port or free_port(state)
    if port not in state["ports"]:
        state["ports"].append(port)
    serve_args = [expand(a, state, port) for a in args.serve_args]
    env = child_env([expand(e, state, port) for e in args.env])
    host = "127.0.0.1"
    cmd = [python(), "-m", "systemone_serve.cli", "serve", "--host", host, "--port", str(port), *serve_args]
    _start(state, name, "serve", port, cmd, env, args.env, args.timeout)


def _start(state: dict, name: str, kind: str, port: int, cmd: list[str], env: dict, env_added: list[str],
           timeout: float, extra: dict | None = None) -> None:
    host = "127.0.0.1"
    base = scratch() / name
    with open(f"{base}.stdout", "wb") as out, open(f"{base}.stderr", "wb") as err:  # the child keeps its own copies
        sup = subprocess.Popen([sys.executable, str(HERE), "_supervise", f"{base}.pid", f"{base}.exit", "--", *cmd],
                               stdout=out, stderr=err, stdin=subprocess.DEVNULL, env=env, start_new_session=True,
                               cwd=str(REPO))
    url = f"http://{host}:{port}"
    inst = {"kind": kind, "pid": None, "supervisor": sup.pid, "port": port, "url": url, "cmd": cmd,
            "env_added": env_added, "base": str(base), "started": time.time(), **(extra or {})}
    deadline = time.monotonic() + timeout
    status = 0
    while time.monotonic() < deadline:
        if inst["pid"] is None and Path(f"{base}.pid").exists():
            inst["pid"] = int(Path(f"{base}.pid").read_text())
        if Path(f"{base}.exit").exists():
            break
        status, _ = http_get(f"{url}/health", timeout=1)
        if status == 200 and inst["pid"]:
            break
        time.sleep(0.1)
    ready = status == 200 and not Path(f"{base}.exit").exists()
    record = {"instance": name, "kind": kind, "command": cmd, "env_added": env_added, "url": url, "ready": ready,
              "seconds": round(time.time() - inst["started"], 2),
              "stdout": Path(f"{base}.stdout").read_text(errors="replace")[-4000:],
              "stderr_tail": Path(f"{base}.stderr").read_text(errors="replace")[-2000:]}
    if Path(f"{base}.exit").exists():
        record["exit_code"] = int(Path(f"{base}.exit").read_text())
    state["instances"][name] = inst
    state["order"].append(name)
    save_state(state)
    write_evidence(f"instances/{name}-launch", record)
    print(json.dumps({k: record[k] for k in ("instance", "url", "ready", "seconds")} |
                     ({"exit_code": record["exit_code"], "stdout": record["stdout"]} if "exit_code" in record else {})))
    if not ready:
        sys.exit(1)


def cmd_tap(args: argparse.Namespace) -> None:
    """A recording reverse proxy in front of instance TO: every request it gets (method, path, headers, body) is
    appended to evidence/instances/NAME.requests.jsonl, then forwarded unchanged and the reply returned unchanged."""
    state = load_state()
    target = state["instances"].get(args.to) or fail(f"no instance {args.to!r} in this run")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name) or args.name in state["instances"]:
        fail(f"instance name {args.name!r} is invalid or already used in this run")
    port = free_port(state)
    log = evidence() / "instances" / f"{args.name}.requests.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch()
    cmd = [sys.executable, str(HERE), "_tap", str(port), target["url"], str(log)]
    _start(state, args.name, "tap", port, cmd, child_env([]), [], args.timeout,
           {"tap_to": args.to, "requests_log": str(log)})


def cmd_tap_server(args: argparse.Namespace) -> None:
    """Internal: the tap itself (standard library only)."""
    import http.server
    import threading

    lock = threading.Lock()
    target, log = args.target.rstrip("/"), Path(args.log)

    class Tap(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _relay(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            if self.path != "/health" and not self.headers.get(PROBE_HEADER):  # the helper's probes are not traffic
                try:
                    shown = json.loads(body) if body else None
                except ValueError:
                    shown = body.decode("utf-8", "replace")
                entry = {"at": datetime.datetime.now().astimezone().isoformat(timespec="milliseconds"),
                         "method": self.command, "path": self.path,
                         "headers": {k.lower(): v for k, v in self.headers.items()}, "body": shown}
                with lock, log.open("a") as out:
                    out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            forward = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
            request = urllib.request.Request(target + self.path, data=body or None, method=self.command, headers=forward)
            try:
                with urllib.request.urlopen(request, timeout=60) as reply:
                    status, data, ctype = reply.status, reply.read(), reply.headers.get("content-type", "")
            except urllib.error.HTTPError as error:
                status, data, ctype = error.code, error.read(), error.headers.get("content-type", "")
            self.send_response(status)
            self.send_header("content-type", ctype or "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = _relay

        def log_message(self, *args) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", int(args.port)), Tap)
    print(json.dumps({"event": "listening", "url": f"http://127.0.0.1:{args.port}", "tap_to": target}), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    server.serve_forever()
    sys.exit(143)


def cmd_requests(args: argparse.Namespace) -> None:
    state = load_state()
    inst = state["instances"].get(args.name) or fail(f"no instance {args.name!r} in this run")
    if inst.get("kind") != "tap":
        fail(f"{args.name} is not a tap; only taps record requests")
    out = [json.loads(line) for line in Path(inst["requests_log"]).read_text().splitlines() if line.strip()]
    results = checks(args.check, {"out": out})
    ok = all(r["ok"] for r in results)
    path = write_evidence(args.step, {"tap": args.name, "requests_seen": len(out), "requests": out[-5:],
                                      "checks": results, "ok": ok})
    print(json.dumps({"step": args.step, "requests_seen": len(out),
                      "last_via": out[-1]["headers"].get("x-systemone-serve-via") if out else None,
                      "ok": ok, "checks": results, "evidence": str(path)}, ensure_ascii=False))
    sys.exit(0 if ok else 1)


def cmd_supervise(args: argparse.Namespace) -> None:
    """Internal: run `sys1 serve` as a child so its exit code can be recorded after the launcher has exited."""
    child = subprocess.Popen(args.cmd)
    Path(args.pidfile).write_text(str(child.pid))
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda *_: None)  # stay alive to record the code; cleanup signals the child directly
    code = child.wait()
    Path(args.exitfile).write_text(str(128 - code if code < 0 else code))


def cmd_doctor(args: argparse.Namespace) -> None:
    state = load_state()
    report = {"run_id": run_id(), "repo": str(REPO), "python": python(), "instances": {}}
    pkg = subprocess.run([python(), "-c", "import systemone_serve as s; print(s.__file__)"], capture_output=True,
                         text=True, env=child_env([]), check=False).stdout.strip()
    report["package_from_repo"] = pkg.startswith(str(REPO / "src"))
    healthy = report["package_from_repo"]
    for name in state["order"]:
        inst = state["instances"][name]
        row: dict = {"url": inst["url"], "pid": inst["pid"]}
        row["alive"] = bool(inst["pid"]) and alive(inst["pid"])
        row["ours"] = row["alive"] and ours(inst)
        row["health"], _ = http_get(f"{inst['url']}/health")
        owner = port_owner(inst["port"])
        row["port_owner"] = "unknown (no ss or lsof)" if owner is None else owner
        row["port_owned_by_us"] = None if owner is None else owner == str(inst["pid"])
        code, text = http_get(f"{inst['url']}/v1/models")
        try:
            row["models"] = [m["name"] for m in json.loads(text)["models"]] if code == 200 else f"HTTP {code}"
        except (ValueError, KeyError):
            row["models"] = text[:200]
        log = Path(f"{inst['base']}.stdout")
        listening = [e for e in _events(log.read_text(errors="replace")) if e.get("event") == "listening"]
        row["listening_event"] = bool(listening) and listening[-1].get("url") == inst["url"]
        row["ok"] = bool(row["ours"] and row["health"] == 200 and row["listening_event"]
                         and row["port_owned_by_us"] is not False)
        healthy = healthy and row["ok"]
        report["instances"][name] = row
    report["ok"] = healthy
    print(json.dumps(report, indent=2))
    sys.exit(0 if healthy else 1)


def _events(text: str) -> list[dict]:
    """sys1 prints indented JSON objects one after another; split them."""
    events, decoder, i = [], json.JSONDecoder(), 0
    while True:
        start = text.find("{", i)
        if start < 0:
            return events
        try:
            obj, i = decoder.raw_decode(text, start)
            events.append(obj)
        except ValueError:
            i = start + 1


def _raw_http(host: str, port: int, method: str, path: str, headers: list[bytes], body: bytes,
              timeout: float) -> tuple[int, dict, bytes]:
    head = [f"{method} {path} HTTP/1.1".encode(), f"Host: {host}:{port}".encode(), b"Connection: close",
            f"Content-Length: {len(body)}".encode(), *headers]
    if body:
        head.append(b"Content-Type: application/json")
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(b"\r\n".join(head) + b"\r\n\r\n" + body)
        data = b""
        while chunk := sock.recv(65536):
            data += chunk
    head_bytes, _, payload = data.partition(b"\r\n\r\n")
    lines = head_bytes.decode("latin-1").split("\r\n")
    status = int(lines[0].split()[1]) if lines and len(lines[0].split()) > 1 else 0
    reply_headers = {k.strip().lower(): v.strip() for k, _, v in (line.partition(":") for line in lines[1:])}
    if reply_headers.get("transfer-encoding") == "chunked":
        decoded, rest = b"", payload
        while rest:
            size_line, _, rest = rest.partition(b"\r\n")
            size = int(size_line.split(b";")[0] or b"0", 16)
            if size == 0:
                break
            decoded, rest = decoded + rest[:size], rest[size + 2:]
        payload = decoded
    return status, reply_headers, payload


def cmd_post(args: argparse.Namespace) -> None:
    state = load_state()
    inst = state["instances"].get(args.name) or fail(f"no instance {args.name!r} in this run")
    if args.body:
        body = json.loads(Path(args.body).read_text()) if args.body != "-" else json.load(sys.stdin)
    else:
        body = json.loads((REPO / "examples" / "support-ticket.json").read_text())
    if args.model and isinstance(body, dict):
        body["model"] = args.model
    payload = b"" if args.method == "GET" else (args.raw_body.encode() if args.raw_body is not None
                                                else json.dumps(body).encode())
    headers = [h.encode("utf-8") for h in args.header]
    if args.via is not None:
        headers.append(b"x-systemone-serve-via: " + args.via.encode("utf-8"))
    if args.via_bytes is not None:
        headers.append(b"x-systemone-serve-via: " + bytes.fromhex(args.via_bytes))
    started = time.monotonic()
    status, reply_headers, raw = _raw_http("127.0.0.1", inst["port"], args.method, args.path, headers, payload,
                                           args.timeout)
    seconds = round(time.monotonic() - started, 3)
    try:
        out = json.loads(raw)
    except ValueError:
        out = raw.decode("utf-8", "replace")
    results = checks(args.check, {"out": out, "status": status})
    ok = (args.expect_status is None or status == args.expect_status) and all(r["ok"] for r in results)
    record = {"instance": args.name, "request": {"method": args.method, "url": inst["url"] + args.path,
                                                 "headers": [repr(h)[2:-1] for h in headers],
                                                 "body": payload.decode("utf-8", "replace")},
              "response": {"status": status, "content_type": reply_headers.get("content-type"), "body": out},
              "seconds": seconds, "expect_status": args.expect_status, "checks": results, "ok": ok}
    path = write_evidence(args.step, record)
    print(json.dumps({"step": args.step, "status": status, "ok": ok, "seconds": seconds, "body": out,
                      "checks": results, "evidence": str(path)}, ensure_ascii=False))
    sys.exit(0 if ok else 1)


def cmd_cli(args: argparse.Namespace) -> None:
    state = load_state()
    port = free_port(state)
    save_state(state)
    argv = [expand(a, state, port) for a in args.sys1_args]
    env = child_env([expand(e, state, port) for e in args.env])
    cmd = [python(), "-m", "systemone_serve.cli", *argv]
    stdin = (REPO / args.stdin).read_bytes() if args.stdin else b""
    started = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, env=env,
                            cwd=str(REPO), start_new_session=True)
    try:
        stdout, stderr = proc.communicate(stdin, timeout=args.timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)  # our own child's group, started just above
        stdout, stderr = proc.communicate()
        code = "timeout"
    seconds = round(time.monotonic() - started, 3)
    text = stdout.decode("utf-8", "replace")
    events = _events(text)
    out = events[-1] if events else text
    results = checks(args.check, {"out": out, "events": events, "code": code,
                                  "stderr": stderr.decode("utf-8", "replace")})
    ok = (args.expect_exit is None or code == args.expect_exit) and all(r["ok"] for r in results)
    record = {"command": ["sys1", *argv], "env_added": [expand(e, state, port) for e in args.env],
              "stdin": args.stdin, "exit_code": code, "seconds": seconds, "stdout": out,
              "stderr": stderr.decode("utf-8", "replace")[-4000:], "expect_exit": args.expect_exit,
              "checks": results, "ok": ok}
    path = write_evidence(args.step, record)
    print(json.dumps({"step": args.step, "exit_code": code, "ok": ok, "seconds": seconds, "checks": results,
                      "evidence": str(path)}, ensure_ascii=False))
    sys.exit(0 if ok else 1)


def cmd_note(args: argparse.Namespace) -> None:
    path = evidence() / "notes.md"
    with path.open("a") as notes:
        notes.write(f"- {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} {' '.join(args.text)}\n")
    print(str(path))


def cmd_cleanup(args: argparse.Namespace) -> None:
    state = load_state()
    report = {}
    for name in reversed(state["order"]):
        inst = state["instances"][name]
        row: dict = {"pid": inst["pid"], "port": inst["port"]}
        exit_file = Path(f"{inst['base']}.exit")
        if inst["pid"] and alive(inst["pid"]) and ours(inst):
            started = time.monotonic()
            os.kill(inst["pid"], signal.SIGTERM)
            while time.monotonic() - started < STOP_WAIT and not exit_file.exists():
                time.sleep(0.1)
            row["sigterm_seconds"] = round(time.monotonic() - started, 2)
            if not exit_file.exists():
                os.killpg(inst["supervisor"], signal.SIGKILL)  # the group this run created for the instance
                row["forced"] = f"SIGKILL after {STOP_WAIT}s"
        elif inst["pid"] and alive(inst["pid"]):
            row["skipped"] = f"pid {inst['pid']} is no longer this run's sys1 serve; not touched"
        time.sleep(0.2)
        row["exit_code"] = int(exit_file.read_text()) if exit_file.exists() else None
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as a new server would; TIME_WAIT is fine
            try:
                sock.bind(("127.0.0.1", inst["port"]))
                row["port_free_after"] = True
            except OSError:
                row["port_free_after"] = False
        dest = evidence() / "instances"
        dest.mkdir(parents=True, exist_ok=True)
        for suffix in ("stdout", "stderr"):
            src = Path(f"{inst['base']}.{suffix}")
            if src.exists():
                shutil.copyfile(src, dest / f"{name}.{suffix}")
        report[name] = row
    write_evidence("cleanup", {"instances": report})
    shutil.rmtree(scratch(), ignore_errors=True)
    files = sorted(str(p.relative_to(evidence())) for p in evidence().rglob("*") if p.is_file())
    print(json.dumps({"stopped": report, "scratch_removed": not scratch().exists(), "evidence": str(evidence()),
                      "evidence_files": files}, indent=2))


def cmd_runs(args: argparse.Namespace) -> None:
    rows = []
    for state_file in sorted(SCRATCH_ROOT.glob("*/state.json")):
        state = json.loads(state_file.read_text())
        live = [n for n, i in state["instances"].items() if i["pid"] and alive(i["pid"])]
        rows.append({"run_id": state_file.parent.name, "live_instances": live})
    print(json.dumps(rows, indent=2))


def cmd_evidence(args: argparse.Namespace) -> None:
    print(str(evidence()))


def main() -> None:
    parser = argparse.ArgumentParser(prog="verify.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="start a run: prints `export VERIFY_RUN_ID=...` for eval").set_defaults(run=cmd_init)
    sub.add_parser("port", help="print a free port reserved for this run").set_defaults(run=cmd_port)
    p = sub.add_parser("launch", help="start `sys1 serve` NAME on 127.0.0.1 and wait for /health")
    p.add_argument("name")
    p.add_argument("--port", type=int)
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--timeout", type=float, default=30.0)
    p.set_defaults(run=cmd_launch)
    p = sub.add_parser("_supervise")  # internal
    p.add_argument("pidfile")
    p.add_argument("exitfile")
    p.set_defaults(run=cmd_supervise)
    p = sub.add_parser("tap", help="start a recording proxy NAME in front of instance TO")
    p.add_argument("name")
    p.add_argument("--to", required=True)
    p.add_argument("--timeout", type=float, default=15.0)
    p.set_defaults(run=cmd_tap)
    p = sub.add_parser("_tap")
    p.add_argument("port")
    p.add_argument("target")
    p.add_argument("log")
    p.set_defaults(run=cmd_tap_server)
    p = sub.add_parser("requests", help="record what a tap has received so far")
    p.add_argument("name")
    p.add_argument("step")
    p.add_argument("--check", action="append", default=[], help="Python expression over out (list of requests)")
    p.set_defaults(run=cmd_requests)
    sub.add_parser("doctor", help="read-only health of this run's instances").set_defaults(run=cmd_doctor)
    p = sub.add_parser("post", help="send one raw HTTP request to an instance and record it")
    p.add_argument("name")
    p.add_argument("step", help="evidence name, feature/step")
    p.add_argument("--model")
    p.add_argument("--method", default="POST", choices=["POST", "GET"])
    p.add_argument("--path", default="/v1/systemone")
    p.add_argument("--body", help="request JSON file (default examples/support-ticket.json), or -")
    p.add_argument("--raw-body", help="send this text as the body, unparsed")
    p.add_argument("--header", action="append", default=[], metavar="'Name: value'")
    p.add_argument("--via", help="x-systemone-serve-via value (UTF-8)")
    p.add_argument("--via-bytes", help="x-systemone-serve-via value as hex bytes, e.g. 636166e9 for caf\\xe9")
    p.add_argument("--expect-status", type=int)
    p.add_argument("--check", action="append", default=[], help="Python expression over out, status")
    p.add_argument("--timeout", type=float, default=30.0)
    p.set_defaults(run=cmd_post)
    p = sub.add_parser("cli", help="run one sys1 command with a scrubbed env and record it")
    p.add_argument("step")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--stdin", help="file (relative to the repo) to feed on stdin")
    p.add_argument("--expect-exit", type=int)
    p.add_argument("--check", action="append", default=[], help="Python expression over out, events, code, stderr")
    p.add_argument("--timeout", type=float, default=30.0)
    p.set_defaults(run=cmd_cli)
    p = sub.add_parser("note", help="append a line to the run's notes.md")
    p.add_argument("text", nargs="+")
    p.set_defaults(run=cmd_note)
    sub.add_parser("cleanup", help="stop this run's instances; keep evidence").set_defaults(run=cmd_cleanup)
    sub.add_parser("runs", help="list runs whose scratch state still exists").set_defaults(run=cmd_runs)
    sub.add_parser("evidence", help="print this run's evidence directory").set_defaults(run=cmd_evidence)
    argv = sys.argv[1:]
    rest: list[str] = []
    if "--" in argv:  # everything after the first `--` goes to sys1 untouched
        cut = argv.index("--")
        argv, rest = argv[:cut], argv[cut + 1:]
    args = parser.parse_args(argv)
    args.serve_args = args.cmd = args.sys1_args = rest
    if args.command in ("cli", "_supervise") and not rest:
        parser.error(f"{args.command} needs `-- <command and arguments>`")
    args.run(args)


if __name__ == "__main__":
    main()
