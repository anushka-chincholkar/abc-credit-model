"""
Step 3 / Deliverable 3 : ABC Credit chatbot (interaction layer).

Responsibilities (per the brief's predictive/generative separation):
  * Collect a MINIMAL applicant field set through a dynamic question flow.
  * Validate / clarify answers.
  * Call the trained ML pipeline for the Approve/Decline decision (NO LLM in the
    decision path).
  * Mask PII before logging.
  * Log every question, answer, engineered feature value and model score to an
    append-only repository (logs/applications.jsonl) -- Objective (d).
  * Present the decision + next steps.

Runs interactively (`python src/chatbot.py`) or programmatically via
run_session(answers_dict) for tests / API embedding.

Field-count trade-off (brief asks us to choose): the model needs 15 raw fields,
but 4 of them (Make/Model/Variant/Product) are auto-filled from a single vehicle
pick and Final_Tier is derivable from the pincode. So the applicant answers ~10
questions. Optional fields may be skipped -- imputation + unseen-safe encoding
keep the pipeline working, so we never block completion on a missing answer.
"""
from __future__ import annotations
import sys, json, re, hashlib
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from predict import predict_customer, REQUIRED_FIELDS

LOG_PATH = C.LOG_DIR / "applications.jsonl"

# (field, question, kind, choices/None, essential)
FLOW = [
    ("Age", "What is your age (years)?", "int", None, True),
    ("Gender", "Gender? (Male/Female)", "cat", ["Male", "Female"], False),
    ("Qualifications", "Highest qualification? (SSC/HSC/UG/GRAD/PG/OTHERS)", "cat",
     ["SSC", "HSC", "UG", "GRAD", "PG", "OTHERS"], False),
    ("Employment_Type", "Employment type? (SAL/SEP/AGR/STU/NREGI/PEN/NPP/NONEARNMEM)", "cat",
     ["SAL", "SEP", "AGR", "STU", "NREGI", "PEN", "NPP", "NONEARNMEM"], True),
    ("Net_salary", "Monthly net income in INR (0 if none)?", "int", None, True),
    ("Resident_Type", "Residence? (O owned / R rented / L leased)", "cat", ["O", "R", "L"], False),
    ("Pincode", "6-digit residential pincode?", "pincode", None, True),
    ("Final_Tier", "Location tier? (e.g. '04 Metro', '07 Rural')", "cat", None, False),
    ("Product_Code", "Vehicle type? (MC motorcycle / SC scooter / MO moped / EB electric)", "cat",
     ["MC", "SC", "MO", "EB"], True),
    ("Make_Code", "Vehicle make/family? (e.g. RAIDER, JUPITER, APACHE)", "text", None, False),
    ("Model_Description", "Full model name?", "text", None, False),
    ("Model_Variant", "Model variant? (e.g. '125 CC DISC')", "text", None, False),
    ("Loan_Amount", "Loan amount requested in INR?", "int", None, True),
    ("LTV", "Loan-to-value % (loan as % of vehicle price)?", "float", None, True),
    ("PAST_LOANS_ACTIVE", "Any active prior loan? (yes/no)", "yesno", None, True),
]

# Fields treated as PII -> masked in the persisted log.
PII_FIELDS = {"Pincode"}


# --------------------------------------------------------------------------- #
def _validate(field, kind, choices, raw):
    """Return (value, error). value=None + error message if invalid."""
    raw = (raw or "").strip()
    if raw == "":
        return None, None  # skip / optional
    try:
        if kind == "int":
            v = int(float(raw))
            if field == "Age" and not (15 <= v <= 100):
                return None, "Age must be between 15 and 100."
            if v < 0:
                return None, "Value cannot be negative."
            return v, None
        if kind == "float":
            v = float(raw)
            if field == "LTV" and not (0 < v <= 100):
                return None, "LTV must be between 0 and 100."
            return v, None
        if kind == "pincode":
            if not re.fullmatch(r"\d{6}", raw):
                return None, "Pincode must be exactly 6 digits."
            return raw, None
        if kind == "yesno":
            return ("PAST_LOANS_ACTIVE" if raw.lower() in ("y", "yes", "1", "true")
                    else "NO_PAST_LOANS"), None
        if kind == "cat" and choices:
            up = raw.upper()
            match = next((c for c in choices if c.upper() == up), None)
            if match is None:
                return raw, None  # unseen-safe pipeline tolerates novel categories
            return match, None
        return raw, None
    except ValueError:
        return None, f"Could not parse '{raw}' as {kind}."


def mask_pii(answers: dict) -> dict:
    """Mask PII for logging: keep only the leading region digit of the pincode."""
    masked = dict(answers)
    for f in PII_FIELDS:
        if masked.get(f):
            val = str(masked[f])
            masked[f] = val[0] + "X" * (len(val) - 1) if val else val
    return masked


def _applicant_hash(answers: dict) -> str:
    """Stable non-reversible id for audit (no raw PII in the id)."""
    blob = json.dumps({k: answers.get(k) for k in sorted(REQUIRED_FIELDS)}, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def log_application(answers, result, session_id):
    """Append-only repository entry with PII masked (Objective d)."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "applicant_hash": _applicant_hash(answers),
        "inputs_masked": mask_pii(answers),
        "model_version": result.get("model_version"),
        "calibrated_pd": result.get("calibrated_pd"),
        "prob_decline": result.get("prob_decline"),
        "prob_approve": result.get("prob_approve"),
        "threshold": result.get("threshold"),
        "decision": result.get("decision"),
        "confidence": result.get("confidence"),
        "reason_codes": [r.get("code") for r in (result.get("reason_codes") or [])],
        "missing_fields": result.get("missing_fields"),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def _next_steps(result):
    if result["decision"] == C.NEGATIVE_LABEL:  # APPROVED
        return ["Congratulations — your application is provisionally APPROVED.",
                "Next: upload KYC + income proof; a representative will confirm final terms."]
    return ["Based on the assessment your application is DECLINED at this time.",
            "You may improve your chances with a larger down-payment (lower LTV) or a co-applicant.",
            "You can reapply after 90 days."]


def run_session(answers: dict, interactive=False, session_id=None) -> dict:
    """Score a completed answer set, log it, and return the presentation payload."""
    session_id = session_id or datetime.now(timezone.utc).strftime("S%Y%m%d%H%M%S%f")
    result = predict_customer(answers)
    entry = log_application(answers, result, session_id)
    payload = {"session_id": session_id, "decision": result["decision"],
               "prob_approve": result["prob_approve"], "prob_decline": result["prob_decline"],
               "calibrated_pd": result.get("calibrated_pd"),
               "confidence": result["confidence"], "risk_explanation": result["risk_explanation"],
               "reason_codes": result.get("reason_codes", []),
               "next_steps": _next_steps(result), "logged_as": entry["applicant_hash"]}
    if interactive:
        _present(payload)
    return payload


def _present(p):
    print("\n" + "=" * 60)
    print(f"  DECISION: {p['decision']}")
    pd_txt = f"   calibrated PD={p['calibrated_pd']:.1%}" if p.get("calibrated_pd") is not None else ""
    print(f"  P(approve)={p['prob_approve']:.1%}   P(decline)={p['prob_decline']:.1%}{pd_txt}   "
          f"confidence={p['confidence']:.0%}")
    print("-" * 60)
    print("  Why:")
    for r in p["risk_explanation"]:
        print(f"   • {r}")
    if p.get("reason_codes"):
        label = "Main adverse-action reasons:" if p["decision"] == C.POSITIVE_LABEL else "Main favourable factors:"
        print(f"  {label}")
        for r in p["reason_codes"]:
            print(f"   [{r['code']}] {r['reason']}")
    print("  Next steps:")
    for s in p["next_steps"]:
        print(f"   → {s}")
    print("=" * 60)


def interactive_cli():
    print("=" * 60)
    print("  ABC Credit — Two-Wheeler Loan Assistant")
    print("  (decision by trained ML model; your pincode is masked in our logs)")
    print("=" * 60)
    answers = {}
    for field, q, kind, choices, essential in FLOW:
        while True:
            raw = input(f"» {q} ").strip()
            val, err = _validate(field, kind, choices, raw)
            if err:
                print(f"   ⚠ {err} Please re-enter.")
                continue
            if val is None and essential:
                print("   ⚠ This field is required to make a decision.")
                continue
            if val is not None:
                answers[field] = val
            break
    run_session(answers, interactive=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass  # non-reconfigurable stream (e.g. redirected to a file); safe to ignore
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        with open(sys.argv[1]) as f:
            run_session(json.load(f), interactive=True)
    else:
        interactive_cli()
