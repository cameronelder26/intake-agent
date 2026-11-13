"""Kill switch for new extractions.

Two ways to engage it, either is enough:
  - environment variable INTAKE_KILL_SWITCH=1 (for a deploy-level stop)
  - a KILL file in the state directory (for an operator at a terminal: `intake kill on`)

The pipeline checks before every document, not once per batch, so a running batch
stops at the next document boundary. Review, approval and release keep working so the
queue can drain.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from .config import Settings

ENV_VAR = "INTAKE_KILL_SWITCH"
_TRUTHY = {"1", "true", "yes", "on"}


def engaged(settings: Settings) -> str | None:
    """Return a reason string if the switch is engaged, else None."""
    if os.environ.get(ENV_VAR, "").strip().lower() in _TRUTHY:
        return f"env {ENV_VAR} is set"
    if settings.kill_file.exists():
        note = settings.kill_file.read_text().strip()
        return f"kill file {settings.kill_file}" + (f": {note}" if note else "")
    return None


def engage(settings: Settings, reason: str, actor: str) -> None:
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    settings.kill_file.write_text(f"{stamp} by {actor}: {reason}\n")


def disengage(settings: Settings) -> bool:
    if settings.kill_file.exists():
        settings.kill_file.unlink()
        return True
    return False
