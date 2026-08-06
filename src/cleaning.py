"""
Step 5 : Reproducible data cleaning.

Design principles
-----------------
* **Stateless & leakage-free.** Nothing here is *fit* on the data and the target
  is never consulted. The exact same transformation runs on the training file
  and on a single applicant at inference time, so train/serve skew is impossible.
* **Idempotent.** Running clean twice yields the same result.
* Every rule maps to a specific finding from 00_profile.py (documented inline).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

# Fields that are short codes -> safe to upper-case so "sal"/"SAL" collapse.
_CODE_COLS = [
    "Qualifications", "Employment_Type", "Product_Code", "Resident_Type",
    "Make_Code", "Model_Variant", "PAST_LOANS_ACTIVE",
]


def _fix_mojibake(s: pd.Series) -> pd.Series:
    """Recover UTF-8 text that was double-encoded as Windows-1252.

    Profiling found 5,028 Model_Description rows containing 'â€“' -- the classic
    signature of an en-dash (U+2013, bytes E2 80 93) decoded as cp1252. Reversing
    that (latin-1 encode -> utf-8 decode) restores the real character. We only
    touch rows that carry the tell-tale 'Ã'/'â€' so clean ASCII is left untouched.
    """
    def repair(x):
        if not isinstance(x, str):
            return x
        if "Ã" in x or "â€" in x or "â" in x:
            try:
                return x.encode("latin-1").decode("utf-8")
            except (UnicodeDecodeError, UnicodeEncodeError):
                return x
        return x
    return s.map(repair)


def clean_dataframe(df: pd.DataFrame, *, drop_null_target: bool = False) -> pd.DataFrame:
    """Apply all deterministic cleaning rules. Returns a new DataFrame."""
    df = df.copy()

    # --- 0. Drop the fully-empty trailing row / any row with no target ------ #
    # Profiling: last CSV line is entirely NaN (ID + target null).
    if drop_null_target and C.TARGET in df.columns:
        df = df[df[C.TARGET].notna()].copy()

    # --- 1. Whitespace hygiene on every text column ------------------------- #
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)   # collapse internal runs of spaces
            .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
        )

    # --- 2. Mojibake repair (before we use the text for features) ---------- #
    if "Model_Description" in df.columns:
        df["Model_Description"] = _fix_mojibake(df["Model_Description"].astype("string"))

    # --- 3. Case normalisation --------------------------------------------- #
    for col in _CODE_COLS:
        if col in df.columns:
            df[col] = df[col].str.upper()
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].str.title()  # Male / Female
    if "Final_Tier" in df.columns:
        # keep the human label but standardise spacing/case of the descriptor
        df["Final_Tier"] = df["Final_Tier"].str.strip()

    # --- 4. Numeric coercion ----------------------------------------------- #
    # Age arrived as object because one value was a lone space -> becomes NaN.
    for col in ["Age", "Loan_Amount", "LTV", "Net_salary"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 5. Domain validity fixes ------------------------------------------ #
    # LTV of 0 (or negative) is impossible: the loan is a % of vehicle value.
    # 65 such rows -> set to NaN so the imputer handles them like other gaps.
    if "LTV" in df.columns:
        df.loc[(df["LTV"] <= C.LTV_MIN) | (df["LTV"] > C.LTV_MAX), "LTV"] = np.nan

    # Age: the brief's policy band is 18-60, but 1,105 rows carry ages 61-69.
    # We do NOT delete them (real applications, and the holdout may contain the
    # same) -- we keep the value and let feature engineering add a policy flag.
    # Guard only against physically impossible ages.
    if "Age" in df.columns:
        df.loc[(df["Age"] < 15) | (df["Age"] > 100), "Age"] = np.nan

    # Pincode -> string (it is an identifier, not a magnitude). Read as float by
    # pandas, so strip the trailing ".0".
    if "Pincode" in df.columns:
        df["Pincode"] = (
            pd.to_numeric(df["Pincode"], errors="coerce")
            .astype("Int64").astype("string")
        )

    # ID -> nullable integer
    if C.ID_COL in df.columns:
        df[C.ID_COL] = pd.to_numeric(df[C.ID_COL], errors="coerce").astype("Int64")

    # --- 6. De-duplicate exact repeats (ignoring ID) ----------------------- #
    # 12 feature-identical rows; harmless to drop and avoids double-weighting.
    if drop_null_target:
        feat_cols = [c for c in df.columns if c != C.ID_COL]
        df = df.drop_duplicates(subset=feat_cols).reset_index(drop=True)

    return df
