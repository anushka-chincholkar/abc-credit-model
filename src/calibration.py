"""
Probability calibration + adverse-action reason codes.

Why
---
The class-weighted CatBoost outputs a good *ranking* but its raw scores are not
true probabilities (weighting inflates them toward 0.5). Regulatory use (IND-AS
109 / ECL provisioning, risk-based pricing, adverse-action notices) needs a
**calibrated Probability of Default (PD)** in real bad-rate space, plus
standardized **reason codes** explaining each decline.

Design
------
* `CalibratedPipeline` wraps the prefit base pipeline with an IsotonicRegression
  map fit on a *held-out* validation set (test stays untouched). Isotonic is
  monotone, so ranking metrics (AUC/PR-AUC/KS) are preserved; only the
  probability axis is corrected.
* Reason codes: per-applicant SHAP contributions (CatBoost native) are aggregated
  from the 93 encoded features up to a small set of business concepts, then ranked.
  The concepts pushing hardest toward DECLINE become the adverse-action reasons.
"""
from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, ClassifierMixin

# feature-name substring -> (reason_code, DIRECTION-NEUTRAL concept label).
# The label names the factor; the per-applicant `direction` ("increases risk" /
# "reduces risk") is computed from the SHAP sign, so the same label reads correctly
# whether the factor helped or hurt (e.g. low LTV -> "reduces risk").
REASON_CATALOG = [
    ("LTV_LOW_EQUITY",   "Loan-to-value / down-payment (collateral cushion)",
     ["ltv", "high_ltv", "down_payment", "vehicle_value"]),
    ("HIGH_LOAN_BURDEN", "Loan size relative to income (repayment burden)",
     ["loan_to_income", "income_to_loan", "loan_amount"]),
    ("INCOME_WEAK",      "Self-reported income level",
     ["net_salary", "is_non_earner", "salary_band"]),
    ("NO_CREDIT_HISTORY","Prior loan / repayment track record",
     ["past_loan", "past_loans_active"]),
    ("GEOGRAPHIC_RISK",  "Location / region",
     ["pincode", "final_tier"]),
    ("AGE_FACTOR",       "Applicant age",
     ["age"]),
    ("EMPLOYMENT_RISK",  "Employment / occupation type",
     ["employment_type"]),
    ("VEHICLE_RISK",     "Vehicle type / make / model",
     ["product_code", "make_code", "model_", "engine_cc"]),
    ("QUALIFICATION",    "Education / qualification",
     ["qualifications"]),
    ("RESIDENCE",        "Residence type",
     ["resident_type"]),
    ("DEMOGRAPHIC",      "Demographic factor",
     ["gender"]),
]


def _concept_for(feat_name: str) -> str | None:
    f = feat_name.lower()
    for code, _msg, keys in REASON_CATALOG:
        if any(k in f for k in keys):
            return code
    return None


_MSG = {code: msg for code, msg, _ in REASON_CATALOG}


class CalibratedPipeline(BaseEstimator, ClassifierMixin):
    """Prefit base pipeline + isotonic PD calibrator + reason-code generator."""

    def __init__(self, base_pipeline, iso: IsotonicRegression, feature_names: list[str]):
        self.base = base_pipeline
        self.iso = iso
        self.feature_names = [f.split("__", 1)[-1] for f in feature_names]
        self.classes_ = np.array([0, 1])

    # -- probabilities ----------------------------------------------------- #
    def raw_proba(self, X):
        return self.base.predict_proba(X)[:, 1]

    def predict_pd(self, X):
        """Calibrated Probability of DECLINE (bad), in real bad-rate space."""
        return self.iso.transform(self.raw_proba(X))

    def predict_proba(self, X):
        pd1 = self.predict_pd(X)
        return np.column_stack([1.0 - pd1, pd1])

    def predict(self, X, threshold=0.5):
        return (self.predict_pd(X) >= threshold).astype(int)

    # -- explanations ------------------------------------------------------ #
    def _transform(self, X):
        return np.asarray(
            self.base.named_steps["pre"].transform(self.base.named_steps["fe"].transform(X)),
            dtype=float,
        )

    def shap_values(self, X):
        """Per-row SHAP contributions (toward the raw DECLINE score)."""
        Xt = self._transform(X)
        model = self.base.named_steps["model"]
        try:
            from catboost import Pool
            sv = model.get_feature_importance(Pool(Xt), type="ShapValues")
            return np.asarray(sv)[:, :-1]  # drop bias column
        except Exception:
            import shap
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(Xt)
            if isinstance(sv, list):
                sv = sv[1]
            return np.asarray(sv)

    def reason_codes(self, X, top=4, favourable=False):
        """Adverse-action (or favourable) reason codes for a single applicant.

        Aggregates SHAP contributions to concept level; returns the concepts
        pushing hardest toward DECLINE (or APPROVE if favourable=True).
        """
        sv = self.shap_values(X)[0]
        agg: dict[str, float] = {}
        for name, contrib in zip(self.feature_names, sv):
            code = _concept_for(name)
            if code is None:
                continue
            agg[code] = agg.get(code, 0.0) + float(contrib)
        if not agg:
            return []
        items = sorted(agg.items(), key=lambda kv: kv[1], reverse=not favourable)
        out = []
        for code, val in items[:top]:
            toward_decline = val > 0
            if favourable and toward_decline:
                continue
            if not favourable and not toward_decline:
                continue
            out.append({
                "code": code,
                "reason": _MSG[code],
                "direction": "increases risk" if toward_decline else "reduces risk",
                "contribution": round(float(val), 4),
            })
        return out
