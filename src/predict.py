"""
Step 20 : Production inference API.

predict_customer(raw_applicant) takes a dict of RAW fields (exactly what the
chatbot collects) and returns probabilities, the final decision, a confidence
score, and a plain-language risk explanation.

The credit decision is produced SOLELY by the trained ML pipeline against the
saved threshold -- no language model is involved in the decision (per the brief's
predictive/generative separation). The explanation is a deterministic, rule-based
narration of the applicant's own risk factors (grounded in the EDA / SHAP drivers)
-- it explains the score, it does not compute it.
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import config as C

# The raw predictor fields the pipeline expects (order irrelevant; extras ignored).
REQUIRED_FIELDS = [
    "Qualifications", "Employment_Type", "Age", "Pincode", "Gender", "Product_Code",
    "Loan_Amount", "LTV", "Resident_Type", "Net_salary", "Final_Tier",
    "Make_Code", "Model_Description", "Model_Variant", "PAST_LOANS_ACTIVE",
]


@lru_cache(maxsize=1)
def _load():
    """Prefer the calibrated model (emits real PDs + reason codes); fall back to
    the raw pipeline if calibration hasn't been run."""
    import calibration  # noqa: F401  (ensures CalibratedPipeline is importable for unpickle)
    cal_path = C.ARTIFACT_DIR / "model_calibrated.joblib"
    with open(C.ARTIFACT_DIR / "model_metadata.json") as f:
        meta = json.load(f)
    if cal_path.exists():
        return joblib.load(cal_path), meta, True
    return joblib.load(C.ARTIFACT_DIR / "model_pipeline.joblib"), meta, False


def _risk_reasons(a: dict, p_decline: float) -> list[str]:
    """Deterministic, EDA-grounded narration of the main risk factors.
    Robust to partial applicants (missing columns) for the API path."""
    reasons = []
    from cleaning import clean_dataframe
    row = clean_dataframe(pd.DataFrame([a]))

    def num(name):  # value or NaN if the column is absent/empty
        if name in row.columns:
            return pd.to_numeric(row[name], errors="coerce").iloc[0]
        return float("nan")

    ltv, loan, sal = num("LTV"), num("Loan_Amount"), num("Net_salary")
    pla = (str(row["PAST_LOANS_ACTIVE"].iloc[0]).upper()
           if "PAST_LOANS_ACTIVE" in row.columns else "")

    if pd.notna(ltv):
        if ltv >= 95:
            reasons.append(f"Very high loan-to-value ({ltv:.0f}%): minimal down-payment, thin collateral cushion (top risk band).")
        elif ltv >= 85:
            reasons.append(f"High loan-to-value ({ltv:.0f}%): below-average equity in the vehicle.")
        elif ltv <= 65:
            reasons.append(f"Healthy loan-to-value ({ltv:.0f}%): sizeable down-payment reduces risk.")
    if pd.notna(loan) and pd.notna(sal) and sal > 0:
        r = loan / sal
        if r >= 4:
            reasons.append(f"Loan is {r:.1f}× monthly income: elevated repayment burden.")
        elif r <= 2:
            reasons.append(f"Loan is only {r:.1f}× monthly income: comfortable repayment burden.")
    if sal == 0:
        reasons.append("No self-reported income (student / non-earning): repayment capacity unverified.")
    if pla == "PAST_LOANS_ACTIVE":
        reasons.append("Has an active prior loan: existing repayment track record (risk-reducing).")
    elif pla == "NO_PAST_LOANS":
        reasons.append("No prior loan on record: no repayment history to lean on.")
    if not reasons:
        reasons.append("Risk profile close to the portfolio average.")
    return reasons


def predict_customer(raw_applicant: dict) -> dict:
    """Score one applicant. Returns a calibrated PD, decision, confidence,
    plain-language risk factors and standardized adverse-action reason codes."""
    model, meta, calibrated = _load()

    missing = [f for f in REQUIRED_FIELDS if f not in raw_applicant or raw_applicant[f] in (None, "")]
    X = pd.DataFrame([{f: raw_applicant.get(f) for f in REQUIRED_FIELDS}])

    if calibrated:
        pd_decline = float(model.predict_pd(X)[0])                     # true PD
        thr = float(meta["calibration"]["calibrated_pd_threshold"])
    else:
        pd_decline = float(model.predict_proba(X)[:, 1][0])
        thr = float(meta["threshold"])

    p_approve = 1.0 - pd_decline
    decision = C.POSITIVE_LABEL if pd_decline >= thr else C.NEGATIVE_LABEL
    confidence = float(min(1.0, abs(pd_decline - thr) / max(thr, 1 - thr)))

    # standardized reason codes (adverse-action if declined, favourable if approved)
    reason_codes = []
    if calibrated:
        try:
            reason_codes = model.reason_codes(X, top=4, favourable=(decision == C.NEGATIVE_LABEL))
        except Exception:
            reason_codes = []

    return {
        "prob_approve": round(p_approve, 4),
        "prob_decline": round(pd_decline, 4),   # calibrated PD of default/decline
        "calibrated_pd": round(pd_decline, 4),
        "decision": decision,
        "threshold": round(thr, 4),
        "confidence": round(confidence, 4),
        "risk_explanation": _risk_reasons(raw_applicant, pd_decline),
        "reason_codes": reason_codes,
        "missing_fields": missing,
        "model_version": meta.get("family"),
        "calibrated": calibrated,
    }


if __name__ == "__main__":
    # smoke example (high-LTV, no income -> should lean risky)
    demo = {
        "Qualifications": "SSC", "Employment_Type": "STU", "Age": 20, "Pincode": "500001",
        "Gender": "Male", "Product_Code": "MC", "Loan_Amount": 130000, "LTV": 96.5,
        "Resident_Type": "R", "Net_salary": 0, "Final_Tier": "07 Rural",
        "Make_Code": "RAIDER", "Model_Description": "TVS RAIDER DISC BSVI",
        "Model_Variant": "RAIDER", "PAST_LOANS_ACTIVE": "NO_PAST_LOANS",
    }
    print(json.dumps(predict_customer(demo), indent=2))
