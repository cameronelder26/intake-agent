"""Runtime settings, read from the environment with sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "claude-haiku-5-5"

# USD per million tokens (input, output). Haiku 5.5 rates apply to prompts under 100K tokens.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    model: str = field(default_factory=lambda: os.environ.get("INTAKE_MODEL", DEFAULT_MODEL))
    effort: str = field(default_factory=lambda: os.environ.get("INTAKE_EFFORT", "low"))
    threshold: float = field(
        default_factory=lambda: float(os.environ.get("INTAKE_THRESHOLD", "0.85"))
    )
    state_dir: Path = field(default_factory=lambda: _env_path("INTAKE_STATE_DIR", ".intake"))
    pg_dsn: str = field(default_factory=lambda: os.environ.get("INTAKE_PG_DSN", ""))

    @property
    def db_path(self) -> Path:
        return self.state_dir / "queue.db"

    @property
    def audit_path(self) -> Path:
        return self.state_dir / "audit.jsonl"

    @property
    def kill_file(self) -> Path:
        return self.state_dir / "KILL"


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token) for mock runs that never call the API."""
    return max(1, round(len(text) / 4)) if text else 0
