from __future__ import annotations

from pathlib import Path

import pytest

from intake_agent.config import Settings
from intake_agent.pipeline import open_state
from intake_agent.synth import generate


@pytest.fixture(scope="session")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Eight synthetic documents covering every template, including one scan."""
    out = tmp_path_factory.mktemp("synthetic")
    generate(out, n=8, seed=11)
    return out


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.delenv("INTAKE_KILL_SWITCH", raising=False)
    return Settings(
        model="claude-haiku-5-5", effort="low", threshold=0.85, state_dir=tmp_path / "state"
    )


@pytest.fixture
def state(settings: Settings):
    return open_state(settings)
