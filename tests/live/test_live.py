"""Live calls. Skipped unless a key or a downloaded snapshot is present."""

import os
from pathlib import Path

import pytest

from systemone_serve.catalog import CATALOG


def _cached(repo: str) -> bool:
    slug = "models--" + repo.replace("/", "--")
    return (Path.home() / ".cache" / "huggingface" / "hub" / slug).is_dir()


@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY is not set")
def test_live_jev() -> None:
    from systemone_serve.door import FrontDoor
    from systemone_serve.types import parse_request

    response = FrontDoor().evaluate(
        parse_request(
            {
                "model": "jev-latest",
                "state": "The refund was denied.",
                "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}},
            }
        )
    )
    assert response.answers["billing"].type == "noul"


@pytest.mark.skipif(
    not any(spec.hub_repo and _cached(spec.hub_repo) for spec in CATALOG),
    reason="no model weights are downloaded",
)
def test_live_weights_present() -> None:
    present = [spec.public_id for spec in CATALOG if spec.hub_repo and _cached(spec.hub_repo)]
    pytest.skip("weights are present but this VM does not load them: " + ", ".join(present))
