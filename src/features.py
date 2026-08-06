"""
Step 6 : Feature engineering.

Implemented as a stateless sklearn transformer so it slots into the Pipeline and
runs identically at train and inference time. It consults only the applicant's
own fields (never the target, never population statistics that would leak), which
keeps it safe inside cross-validation and safe for a single-row prediction.

Every engineered feature carries a one-line business justification.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import config as C

# Column groups this transformer produces (referenced by the ColumnTransformer)
ENGINEERED_NUMERIC = [
    "income_to_loan",     # repayment capacity: monthly income / loan size
    "loan_to_income",     # leverage / burden: loan size / monthly income
    "vehicle_value_est",  # reconstructed asset value = Loan_Amount / (LTV/100)
    "down_payment_est",   # skin in the game = vehicle value - loan
    "engine_cc",          # vehicle displacement parsed from the model text
]
ENGINEERED_CAT = ["age_band", "ltv_band", "salary_band", "pincode_region"]
ENGINEERED_BINARY = [
    "is_non_earner",       # Net_salary == 0 (student / non-earning member)
    "age_outside_policy",  # age beyond the stated 18-60 lending band
    "high_ltv",            # LTV >= 85%, the risk-heavy tail
    "past_loan_active",    # PAST_LOANS_ACTIVE mapped to 0/1
]

# Fixed, data-independent bin edges (justified by the quantiles seen in profiling).
# Hard-coded edges keep the transform stateless and reproducible on unseen data.
_AGE_BINS = [0, 25, 35, 45, 60, 200]
_AGE_LABELS = ["<25", "25-35", "35-45", "45-60", "60+"]
_LTV_BINS = [-1, 60, 75, 85, 95, 1000]
_LTV_LABELS = ["<=60", "60-75", "75-85", "85-95", "95+"]
_SAL_BINS = [-1, 0, 20000, 30000, 45000, 75000, 1e12]
_SAL_LABELS = ["none", "<=20k", "20-30k", "30-45k", "45-75k", "75k+"]

_CC_RE = re.compile(r"(\d{2,4})\s*CC")
_NUM_RE = re.compile(r"(\d{2,4})")


def _parse_engine_cc(variant: object, desc: object) -> float:
    """Extract engine displacement (cc). Electric bikes -> 0. Unknown -> NaN."""
    for text in (variant, desc):
        if not isinstance(text, str):
            continue
        t = text.upper()
        if "EBIKE" in t or "IQUBE" in t or "ELECTRIC" in t:
            return 0.0
        m = _CC_RE.search(t) or _NUM_RE.search(t)
        if m:
            return float(m.group(1))
    return np.nan


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add engineered columns; leave originals in place for the ColumnTransformer."""

    def fit(self, X, y=None):
        return self  # stateless

    def _num(self, df, name):
        """Numeric Series for a column, or all-NaN if the column is absent
        (keeps the transform robust to dropped/skipped fields)."""
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype="float64")

    def _raw(self, df, name):
        if name in df.columns:
            return df[name]
        return pd.Series(pd.NA, index=df.index, dtype="object")

    def transform(self, X):
        df = X.copy()

        salary = self._num(df, "Net_salary")
        loan = self._num(df, "Loan_Amount")
        ltv = self._num(df, "LTV")
        age = self._num(df, "Age")

        # --- capacity / leverage ratios ------------------------------------ #
        # +1 guards against divide-by-zero for non-earners; inf -> NaN (imputed).
        df["income_to_loan"] = (salary / loan).replace([np.inf, -np.inf], np.nan)
        df["loan_to_income"] = (loan / (salary + 1)).replace([np.inf, -np.inf], np.nan)

        # --- reconstructed vehicle economics ------------------------------- #
        # LTV = loan / vehicle_value * 100  =>  value = loan / (LTV/100).
        # A bigger down-payment (skin in the game) usually means lower risk.
        value = (loan / (ltv / 100.0)).replace([np.inf, -np.inf], np.nan)
        df["vehicle_value_est"] = value
        df["down_payment_est"] = value - loan

        # --- vehicle displacement ------------------------------------------ #
        df["engine_cc"] = [
            _parse_engine_cc(v, d)
            for v, d in zip(self._raw(df, "Model_Variant"), self._raw(df, "Model_Description"))
        ]

        # --- binary risk flags --------------------------------------------- #
        df["is_non_earner"] = (salary.fillna(-1) == 0).astype(int)
        df["age_outside_policy"] = (
            ((age < C.AGE_MIN) | (age > C.AGE_MAX)).fillna(False).astype(int)
        )
        df["high_ltv"] = (ltv >= 85).fillna(False).astype(int)
        pla = self._raw(df, "PAST_LOANS_ACTIVE")
        df["past_loan_active"] = (
            pla.astype("string").str.upper().eq("PAST_LOANS_ACTIVE").fillna(False).astype(int)
        )

        # --- interpretable bands (help linear models; requested in brief) --- #
        df["age_band"] = pd.cut(age, bins=_AGE_BINS, labels=_AGE_LABELS).astype("string")
        df["ltv_band"] = pd.cut(ltv, bins=_LTV_BINS, labels=_LTV_LABELS).astype("string")
        df["salary_band"] = pd.cut(salary, bins=_SAL_BINS, labels=_SAL_LABELS).astype("string")

        # --- geography: India postal zone = leading digit of the pincode ---- #
        # 1st digit maps to a broad region (unseen-safe, ~9 levels vs 14k pincodes).
        pin = self._raw(df, "Pincode").astype("string")
        df["pincode_region"] = pin.str.replace(r"\.0$", "", regex=True).str[0]
        df["pincode_region"] = df["pincode_region"].where(
            df["pincode_region"].str.match(r"^[1-9]$").fillna(False), other=pd.NA
        )

        # fill NaN band labels so the OHE sees an explicit "missing" category
        for col in ENGINEERED_CAT:
            df[col] = df[col].fillna("missing")

        # sklearn's SimpleImputer/encoders can't reason about pandas' StringDtype
        # sentinel pd.NA ("boolean value of NA is ambiguous"). Cast every text
        # column to plain numpy object with np.nan so imputers behave.
        str_cols = df.select_dtypes(include=["string", "object"]).columns
        for col in str_cols:
            df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

        return df


# Explicit interaction terms (mostly help linear models; GBMs already learn
# interactions via tree splits -- kept behind a subclass so the experiment can
# quantify whether they add anything).
INTERACTION_NUMERIC = ["ltv_x_loan", "ltv_sq", "ltv_x_nonearner", "burden_x_highltv"]


class InteractionEngineer(FeatureEngineer):
    """FeatureEngineer + a handful of interaction terms."""

    def transform(self, X):
        df = super().transform(X)
        ltv = self._num(df, "LTV")
        loan = self._num(df, "Loan_Amount")
        df["ltv_x_loan"] = (ltv * loan) / 1e6
        df["ltv_sq"] = ltv ** 2
        df["ltv_x_nonearner"] = ltv * df["is_non_earner"]
        df["burden_x_highltv"] = pd.to_numeric(df["loan_to_income"], errors="coerce") * df["high_ltv"]
        return df
