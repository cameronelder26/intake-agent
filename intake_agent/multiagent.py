"""Two agents: an extractor and a reviewer that challenges low-confidence fields.

    extract -> validate -> reviewer -> (gate) -> auto_accept  -> audit
                                              -> queue_review -> human_review (interrupt) -> audit

The extractor agent is the existing Extractor. The reviewer agent looks at each field
that fell below the confidence threshold and decides: "clear" (it checked the value
against the source text and the validators and it holds up) or "escalate" (it does
not, so the field goes to the human queue with the reviewer's reason attached).

The reviewer works through two tools, the same two for the mock and the real model:

    lookup_source_span(field)  where, if anywhere, the extracted value is written in the source
    validate_field(field)      the deterministic validators (NPI checksum, dates, code formats)

MockReviewer is a fixed policy over those tools. It is deterministic and needs no key.
ModelReviewer gives the same tools to a Claude model and is used only when
ANTHROPIC_API_KEY is set. A field the reviewer cannot decide on always escalates.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Settings
from .graph import build_graph, find_citation
from .schema import FIELDS, ScoredField
from .validators import validate

TOOLS: list[dict[str, Any]] = [
    {
        "name": "lookup_source_span",
        "description": (
            "Find where the extracted value of a field is written in the source document. "
            "Returns found=false if the value does not appear in the text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"field": {"type": "string", "enum": list(FIELDS)}},
            "required": ["field"],
        },
    },
    {
        "name": "validate_field",
        "description": "Run the deterministic validators on the extracted value of a field.",
        "input_schema": {
            "type": "object",
            "properties": {"field": {"type": "string", "enum": list(FIELDS)}},
            "required": ["field"],
        },
    },
    {
        "name": "submit_verdict",
        "description": "Give the final decision for one field: clear it or escalate it to a human.",
        "input_schema": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "enum": list(FIELDS)},
                "decision": {"type": "string", "enum": ["clear", "escalate"]},
                "reason": {"type": "string"},
            },
            "required": ["field", "decision", "reason"],
        },
    },
]


@dataclass
class Verdict:
    field: str
    decision: str  # "clear" or "escalate"
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"field": self.field, "decision": self.decision, "reason": self.reason}


class ReviewTools:
    """The reviewer's tools, bound to one document."""

    def __init__(self, scored: dict[str, ScoredField], source_text: str):
        self.scored = scored
        self.text = source_text

    def lookup_source_span(self, field: str) -> dict[str, Any]:
        value = self.scored[field].value
        if value in (None, "", []):
            return {"found": False, "note": "no extracted value to look up"}
        cite = find_citation(value, self.text)
        return {"found": True, **cite} if cite else {"found": False}

    def validate_field(self, field: str) -> dict[str, Any]:
        issues = validate(field, self.scored[field].value)
        return {"valid": not issues, "issues": issues}

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in ("lookup_source_span", "validate_field"):
            return {"error": f"unknown tool {name}"}
        if args.get("field") not in self.scored:
            return {"error": f"unknown field {args.get('field')!r}"}
        return getattr(self, name)(args["field"])


class Reviewer(Protocol):
    name: str

    def review(
        self, scored: dict[str, ScoredField], flagged: list[str], text: str
    ) -> list[Verdict]: ...


class MockReviewer:
    """Deterministic policy over the two tools. No model, no key."""

    name = "mock-reviewer"

    def review(self, scored, flagged, text):
        tools = ReviewTools(scored, text)
        out = []
        for field in flagged:
            check = tools.call("validate_field", {"field": field})
            span = tools.call("lookup_source_span", {"field": field})
            if not check["valid"]:
                out.append(Verdict(field, "escalate", "validator: " + "; ".join(check["issues"])))
            elif not span["found"]:
                out.append(Verdict(field, "escalate", "value not found in the source text"))
            else:
                out.append(
                    Verdict(field, "clear", f"validators pass; source says {span['quote']!r}")
                )
        return out


class ModelReviewer:
    """The same tools given to a Claude model. Used only when ANTHROPIC_API_KEY is set."""

    MAX_TURNS = 6
    SYSTEM = (
        "You review one extracted field from a healthcare intake document. Use "
        "lookup_source_span and validate_field to check it, then call submit_verdict. "
        "Clear the field only if the value is written in the source and passes validation. "
        "If you are unsure, escalate."
    )

    def __init__(self, settings: Settings, client: Any = None):
        import anthropic

        self.model = settings.model
        self.name = f"model-reviewer:{settings.model}"
        self.client = client or anthropic.Anthropic()

    def _one(self, field: str, tools: ReviewTools) -> Verdict:
        s = tools.scored[field]
        msgs: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": f"Field {field!r}: value {json.dumps(s.value)}, "
                f"confidence {s.confidence}, known issues {s.issues}.",
            }
        ]
        for _ in range(self.MAX_TURNS):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.SYSTEM,
                tools=TOOLS,
                messages=msgs,
            )
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                break
            msgs.append({"role": "assistant", "content": resp.content})
            results = []
            for u in uses:
                if u.name == "submit_verdict":
                    d = u.input.get("decision")
                    return Verdict(
                        field,
                        d if d in ("clear", "escalate") else "escalate",
                        u.input.get("reason", ""),
                    )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": u.id,
                        "content": json.dumps(tools.call(u.name, dict(u.input))),
                    }
                )
            msgs.append({"role": "user", "content": results})
        return Verdict(field, "escalate", "reviewer gave no verdict")

    def review(self, scored, flagged, text):
        tools = ReviewTools(scored, text)
        return [self._one(f, tools) for f in flagged]


def make_reviewer(mock: bool | None, settings: Settings) -> Reviewer:
    if mock is None:
        mock = not os.environ.get("ANTHROPIC_API_KEY")
    return MockReviewer() if mock else ModelReviewer(settings)


def build_multiagent_graph(
    extractor, queue, settings, reviewer: Reviewer | None = None, checkpointer=None, audit=None
):
    return build_graph(
        extractor,
        queue,
        settings,
        audit=audit,
        reviewer=reviewer or MockReviewer(),
        checkpointer=checkpointer,
    )
