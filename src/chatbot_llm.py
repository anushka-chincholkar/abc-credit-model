"""
LLM-driven ABC Credit chatbot (interaction layer) over the deployed ML model.

Architecture (predictive/generative separation — the brief's hard rule):
  * The ORCHESTRATOR (this file, plain Python) owns the conversation flow.
  * Claude Haiku 4.5 does ONLY per-answer natural-language normalisation and the
    final explanation (via llm_client). It never decides.
  * The DECISION is computed by predict_customer() (the calibrated ML model).
  * Fields are DERIVED where possible: one vehicle pick fills 4 fields, price+loan
    give LTV, the pincode gives the tier — so the applicant answers ~9 questions.

Runs interactively (`python src/chatbot_llm.py`), scripted (run_session(answers)),
and fully offline (no ANTHROPIC_API_KEY -> deterministic normalisation + template
explanation), so it is testable without a key.
"""
from __future__ import annotations
import sys, json, re
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from predict import predict_customer
from chatbot import _validate, mask_pii, log_application, _next_steps
import llm_client

_ASSETS = {}
def _assets():
    if not _ASSETS:
        _ASSETS["veh"] = json.loads((C.ARTIFACT_DIR / "vehicle_catalog.json").read_text())
        _ASSETS["pin"] = json.loads((C.ARTIFACT_DIR / "pincode_tier.json").read_text())
        rt_path = C.ARTIFACT_DIR / "region_tier.json"
        _ASSETS["region"] = json.loads(rt_path.read_text()) if rt_path.exists() else {}
    return _ASSETS


def _tier_for(pincode: str):
    """Exact pincode -> tier; else region-digit fallback; else global default."""
    a = _assets()
    pin, region = a["pin"], a.get("region", {})
    s = str(pincode)
    return pin.get(s) or region.get(s[:1]) or region.get("_default")


# Conversation flow — high-signal fields first. `kind` drives validation; `derive`
# marks steps that fill several model fields. Order = what the applicant is asked.
STEPS = [
    ("vehicle", "Which two-wheeler are you financing?", "vehicle", None, True),
    ("vehicle_price", "What is the on-road price of the vehicle (₹)?", "int", None, True),
    ("Loan_Amount",   "How much loan do you need (₹)?", "int", None, True),
    ("Pincode",       "Your 6-digit residential pincode?", "pincode", None, True),
    ("Employment_Type", "Employment type? (SAL salaried / SEP self-employed / AGR agriculture / STU student / NREGI / PEN / NPP / NONEARNMEM)", "cat",
                        ["SAL", "SEP", "AGR", "STU", "NREGI", "PEN", "NPP", "NONEARNMEM"], True),
    ("Net_salary",    "Monthly income in ₹ (enter 0 if none)?", "int", None, True),
    ("PAST_LOANS_ACTIVE", "Do you currently have an active loan? (yes/no)", "yesno", None, True),
    ("Age",           "Your age (years)?", "int", None, True),
    ("Gender",        "Gender? (Male/Female) — optional", "cat", ["Male", "Female"], False),
    ("Qualifications", "Highest qualification? (SSC/HSC/UG/GRAD/PG/OTHERS) — optional", "cat",
                       ["SSC", "HSC", "UG", "GRAD", "PG", "OTHERS"], False),
    ("Resident_Type", "Residence? (O owned / R rented / L leased) — optional", "cat", ["O", "R", "L"], False),
]

import difflib

# Below this similarity, we no longer trust the match — treat the vehicle as
# genuinely new/unknown rather than forcing it into a catalogue entry.
_VEHICLE_FUZZY_CUTOFF = 0.6


def _match_vehicle(text: str) -> str | None:
    """Best-effort map of free text / index to a catalog Model_Description.
    Returns None when nothing in the catalogue is a close enough match —
    the caller then keeps the applicant's own wording instead of blocking."""
    veh = _assets()["veh"]; popular = veh["popular"]; catalog = veh["catalog"]
    t = (text or "").strip()
    if not t:
        return None
    if t.isdigit():
        i = int(t) - 1
        return popular[i] if 0 <= i < len(popular) else None
    up = t.upper()
    for name in catalog:                          # exact match
        if up == name.upper():
            return name
    hits = [name for name in catalog if up in name.upper()]   # substring
    if not hits:
        hits = [name for name in catalog
                if any(w in name.upper() for w in up.split() if len(w) > 2)]
    if hits:
        return max(hits, key=lambda n: catalog[n]["count"])
    close = difflib.get_close_matches(                        # fuzzy fallback
        up, [n.upper() for n in catalog], n=1, cutoff=_VEHICLE_FUZZY_CUTOFF)
    if close:
        for name in catalog:
            if name.upper() == close[0]:
                return name
    return None  # no close match -> genuinely new/unrecognised model


def _guess_product_code(desc: str) -> str:
    """Coarse fallback Product_Code for a model with no catalogue match."""
    t = (desc or "").upper()
    if any(k in t for k in ("EBIKE", "ELECTRIC", "IQUBE", " EV")):
        return "EB"
    if any(k in t for k in ("SCOOT", "SCOOTY", "ACTIVA", "JUPITER")):
        return "SC"
    if "MOPED" in t:
        return "MO"
    return "MC"

def _collect_value(step, raw_text):
    """Normalise (LLM or offline) then validate. Returns (value, error)."""
    key, prompt, kind, choices, _ = step
    if kind == "vehicle":
        norm = raw_text
        if llm_client.online():
            norm = llm_client.normalize_answer(
                "vehicle", "vehicle", _assets()["veh"]["popular"], prompt, raw_text) or raw_text
        match = _match_vehicle(norm) or _match_vehicle(raw_text)
        if match:
            return match, None
        # Not close to anything in the catalogue -> keep it as a new model
        # rather than blocking. Prediction is never gated on a catalog hit.
        cleaned = (norm or raw_text).strip()
        return (cleaned or None), (None if cleaned else "Which vehicle are you financing?")
    norm = llm_client.normalize_answer(key, kind, choices, prompt, raw_text)
    return _validate(key, kind, choices, norm if norm is not None else "")

def _assemble(collected: dict) -> dict:
    """Derive the full 15-field applicant record from the collected answers."""
    veh = _assets()["veh"]["catalog"]
    a = {}
    desc = collected.get("vehicle")
    if desc and desc in veh:
        a["Model_Description"] = desc
        a["Make_Code"] = veh[desc]["Make_Code"]
        a["Model_Variant"] = veh[desc]["Model_Variant"]
        a["Product_Code"] = veh[desc]["Product_Code"]
    elif desc:
        # Genuinely new / unrecognised model — keep the applicant's own text.
        # pipeline.py's OneHotEncoder(handle_unknown="infrequent_if_exist") and
        # TargetEncoder are unseen-safe, so the prediction call proceeds
        # normally without ever needing a catalogue match.
        a["Model_Description"] = desc
        a["Make_Code"] = (desc.split()[0].upper() if desc.split() else "OTHER")
        a["Model_Variant"] = desc
        a["Product_Code"] = _guess_product_code(desc)
    a["Loan_Amount"] = collected.get("Loan_Amount")
    price = collected.get("vehicle_price")
    if price and a.get("Loan_Amount") and price > 0:
        a["LTV"] = round(100.0 * a["Loan_Amount"] / price, 2)
    a["Pincode"] = collected.get("Pincode")
    if a.get("Pincode"):
        a["Final_Tier"] = _tier_for(a["Pincode"])
    for f in ["Employment_Type", "Net_salary", "PAST_LOANS_ACTIVE", "Age",
              "Gender", "Qualifications", "Resident_Type"]:
        if f in collected:
            a[f] = collected[f]
    return a


def _template_explanation(result: dict) -> str:
    pd_pct = f"{result['calibrated_pd']*100:.0f}%" if result.get("calibrated_pd") is not None else "n/a"
    lines = [f"Decision: {result['decision']} (estimated risk / PD {pd_pct})."]
    if result.get("reason_codes"):
        label = "Main favourable factors" if result["decision"] == "APPROVED" else "Main risk factors"
        lines.append(f"{label}: "
                     + "; ".join(f"{r['reason']} ({r['direction']})" for r in result["reason_codes"][:3]) + ".")
    return "\n".join(lines)

def _ltv_policy_flag(applicant: dict) -> str | None:
    ltv = applicant.get("LTV")
    if ltv is not None and ltv > C.LTV_POLICY_CAP:
        return (f"Heads up — the loan amount you're asking for is {ltv:.1f}% of the "
                f"vehicle's price, above our usual {C.LTV_POLICY_CAP:.0f}% financing cap. "
                f"You could lower the loan amount or add a bigger down payment if you'd "
                f"like to adjust it, or I can go ahead and submit it as is.")
    return None

def run_session(answers: dict, interactive: bool = False, session_id: str | None = None) -> dict:
    """Drive the flow from a dict of raw text answers (keys = STEP keys)."""
    session_id = session_id or datetime.now(timezone.utc).strftime("S%Y%m%d%H%M%S%f")
    collected = {}
    for step in STEPS:
        key, prompt, kind, choices, essential = step
        raw = answers.get(key, "")
        value, err = _collect_value(step, raw)
        if value is None:
            if essential:
                raise ValueError(f"Missing/invalid essential field '{key}': {err or raw!r}")
            continue
        collected[key] = value

    applicant = _assemble(collected)
    ltv_flag = _ltv_policy_flag(applicant)
    result = predict_customer(applicant)                    # <-- DECISION (ML, not LLM)

    explanation = llm_client.explain(result) or _template_explanation(result)
    entry = log_application(applicant, result, session_id)   # PII-masked repository log

    payload = {
        "session_id": session_id, "decision": result["decision"],
        "calibrated_pd": result.get("calibrated_pd"),
        "prob_approve": result["prob_approve"], "prob_decline": result["prob_decline"],
        "confidence": result["confidence"], "reason_codes": result.get("reason_codes", []),
        "explanation": explanation, "next_steps": _next_steps(result),
        "derived_fields": {k: applicant.get(k) for k in ("LTV", "Final_Tier", "Make_Code", "Product_Code")},
        "policy_flags": [ltv_flag] if ltv_flag else [],
        "llm_mode": "claude-haiku-4-5" if llm_client.online() else "offline-deterministic",
        "logged_as": entry["applicant_hash"],
    }
    if interactive:
        _present(payload)
    return payload


def _present(p):
    print("\n" + "=" * 64)
    print(f"  DECISION: {p['decision']}   (P(decline)={p['prob_decline']:.1%}, "
          f"calibrated PD={p['calibrated_pd']:.1%}, confidence={p['confidence']:.0%})")
    if p.get("policy_flags"):
        print("-" * 64)
        for f in p["policy_flags"]:
            print(f"  ⚠ {f}") 
    print("-" * 64)
    print(p["explanation"])
    if p["reason_codes"]:
        print("\n  Key factors:")
        for r in p["reason_codes"]:
            print(f"   • [{r['code']}] {r['reason']} — {r['direction']}")
    print("\n  Next steps:")
    for s in p["next_steps"]:
        print(f"   → {s}")
    print(f"\n  (decision by ML model; interaction layer = {p['llm_mode']}; logged {p['logged_as']})")
    print("=" * 64)


def interactive_cli():
    print("=" * 64)
    print("  ABC Credit — Two-Wheeler Loan Assistant")
    print(f"  interaction layer: {'Claude Haiku 4.5' if llm_client.online() else 'offline (no API key)'}")
    print("  (decision made by the trained ML model; your pincode is masked in logs)")
    print("=" * 64)
    veh = _assets()["veh"]["popular"]
    print("\nPopular vehicles:")
    for i, name in enumerate(veh[:15], 1):
        print(f"  {i:2d}. {name}")
    print("  (or type any model name)")

    answers = {}
    for step in STEPS:
        key, prompt, kind, choices, essential = step
        while True:
            raw = input(f"\n» {prompt}\n> ").strip()
            value, err = _collect_value(step, raw)
            if value is None and essential:
                print(f"   ⚠ {err or 'Please try again.'}")
                continue
            if value is not None:
                answers[key] = raw  # store raw; run_session re-parses uniformly
            break
    run_session(answers, interactive=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass  # non-reconfigurable stream (e.g. redirected to a file); safe to ignore
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        run_session(json.load(open(sys.argv[1])), interactive=True)
    else:
        interactive_cli()
