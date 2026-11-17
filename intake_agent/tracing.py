"""LangSmith tracing, switched on by environment variables only.

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=<key>

LangGraph runs are traced by the langsmith package when
both are set, so there is nothing to wire per call. This module only makes the off
state certain: if tracing is requested without a key (or the key is set but tracing is
not requested) nothing is sent and nothing errors. CI sets neither, so it stays offline.
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}


def requested() -> bool:
    return any(
        os.environ.get(v, "").strip().lower() in _TRUTHY
        for v in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
    )


def enabled() -> bool:
    return requested() and bool(os.environ.get("LANGSMITH_API_KEY"))


def configure() -> bool:
    """Make the env consistent. Returns True if runs will be sent to LangSmith."""
    if requested() and not os.environ.get("LANGSMITH_API_KEY"):
        # Asked for tracing but there is no key: turn it off so nothing tries to upload.
        for v in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
            os.environ[v] = "false"
    return enabled()
