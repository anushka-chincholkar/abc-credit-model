"""
Claude interaction-layer client (Haiku 4.5) with a deterministic offline fallback.

Boundary: this module ONLY does natural-language work in the interaction layer —
(1) normalising a free-text answer into a canonical field value, and
(2) turning a decision + reason codes into friendly prose.
It NEVER decides approve/decline. The decision is computed in predict_customer().

If ANTHROPIC_API_KEY is set, calls Claude Haiku 4.5. Otherwise every method falls
back to a deterministic rule-based path, so the whole chatbot is testable offline.
"""
from __future__ import annotations
import os, json
from functools import lru_cache

MODEL = "claude-haiku-4-5"

# System prompts encode the predictive/generative separation (guardrails).
SLOT_SYSTEM = (
    "You are the input-collection layer of a loan-application assistant for an "
    "Indian two-wheeler lender. Your ONLY job is to convert the applicant's free-text "
    "answer into a single canonical value for one specific field. You do NOT decide "
    "whether the loan is approved or declined — a separate system does that. "
    "Return strict JSON only."
)
EXPLAIN_SYSTEM = (
    "You are the communication layer of a loan-application assistant. A separate "
    "machine-learning model has ALREADY made the decision; you are given that decision "
    "and the exact risk factors behind it. Explain the outcome to the applicant in 2-4 "
    "warm, plain-language sentences, then list the next steps. "
    "CRITICAL RULES: (1) Never say the decision was made by you or by an AI — it was made "
    "by the lender's scoring system. (2) Use ONLY the reason codes provided; never invent "
    "or speculate about a reason. (3) Never promise approval or reversal. Return plain text."
)


@lru_cache(maxsize=1)
def _client():
    import anthropic
    return anthropic.Anthropic()


def online() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# --------------------------------------------------------------------------- #
def normalize_answer(field: str, kind: str, choices, prompt: str, user_text: str) -> str | None:
    """LLM: map free text -> a canonical string for `field` (validated downstream).
    Offline: return the raw text (the deterministic validator handles coercion)."""
    if not online():
        return user_text
    choice_txt = f" Allowed values: {choices}." if choices else ""
    schema = {
        "type": "object",
        "properties": {
            "value": {"type": ["string", "null"],
                      "description": "canonical value, or null if the answer is unclear/empty"},
        },
        "required": ["value"],
        "additionalProperties": False,
    }
    user = (f"Field: {field} ({kind}).{choice_txt}\n"
            f"Question asked: {prompt}\n"
            f"Applicant answer: {user_text!r}\n"
            f"Extract the canonical value. For amounts return digits only (e.g. 'thirty "
            f"thousand' -> '30000'). For yes/no return 'yes' or 'no'. For a category, "
            f"return the closest allowed value. If unclear, value=null.")
    try:
        resp = _client().messages.create(
            model=MODEL, max_tokens=128, system=SLOT_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": user}],
        )
        txt = next(b.text for b in resp.content if b.type == "text")
        return json.loads(txt).get("value")
    except Exception:
        # any SDK/model hiccup -> fall back to raw text; validator will handle it
        return user_text


def explain(payload: dict) -> str | None:
    """LLM: render decision + reason codes into friendly prose. Offline: None
    (caller uses a deterministic template)."""
    if not online():
        return None
    facts = {
        "decision": payload["decision"],
        "estimated_probability_of_decline": payload.get("calibrated_pd"),
        "reason_codes": payload.get("reason_codes", []),
        "risk_notes": payload.get("risk_explanation", []),
    }
    try:
        resp = _client().messages.create(
            model=MODEL, max_tokens=400, system=EXPLAIN_SYSTEM,
            messages=[{"role": "user", "content":
                       "Decision facts (JSON):\n" + json.dumps(facts, indent=2) +
                       "\n\nWrite the applicant-facing explanation and next steps."}],
        )
        return next(b.text for b in resp.content if b.type == "text").strip()
    except Exception:
        return None
