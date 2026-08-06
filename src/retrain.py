"""
New-data integration scaffolding (Bucket B — the real ceiling-breakers).

The current ceiling is *data*, not modelling: the 15 application fields top out at
ROC-AUC ~0.78. The two step-change levers both need data we don't yet have. This
module makes them **plug-and-play** so nothing has to be re-architected when the
data arrives:

  1. `bureau_score` feature slot — the single biggest lever in Indian retail
     lending (CIBIL/Experian). The pipeline already tolerates it: add the column
     and register it via `enable_bureau_score()`; absent, everything is unchanged.
  2. Reject inference — today's label `Decision` is the *historical underwriting
     decision*, a proxy. A true PD model needs realised default (e.g. 90+ DPD in
     12 months) on *booked* loans, with the rejected population's outcomes inferred
     so the model isn't biased by the accept-only sample. `retrain_with_default_label`
     shows the exact hook.

Nothing here fabricates signal; these are documented, runnable integration points.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config as C
from pipeline import build_full_pipeline, build_preprocessor, NUMERIC


# --------------------------------------------------------------------------- #
# 1. Bureau-score feature slot
# --------------------------------------------------------------------------- #
def enable_bureau_score(colname: str = "bureau_score"):
    """Register a bureau score as an extra numeric feature.

    Usage once your data has a `bureau_score` column (e.g. CIBIL 300-900):
        from retrain import enable_bureau_score
        pipe = build_full_pipeline(model, extra_numeric=enable_bureau_score())
    The median imputer + RobustScaler already handle range/missing; thin-file
    applicants with no score are imputed to the population median. No other change
    is needed — `build_preprocessor(extra_numeric=[...])` slots it into the numeric
    branch, so the encoder/OHE/target-encoder logic is untouched.
    """
    return [colname]


def bureau_expected_lift_note() -> str:
    return ("Industry benchmark: adding a bureau score typically lifts application "
            "scorecards from Gini ~0.35-0.50 to ~0.55-0.70. On this dataset the "
            "application-only model already reaches Gini 0.556; a bureau score would "
            "most plausibly push it toward or beyond the top of the industry band.")


# --------------------------------------------------------------------------- #
# 2. Realised-default target + reject inference
# --------------------------------------------------------------------------- #
def make_default_label(df: pd.DataFrame, dpd_col: str = "max_dpd_12m",
                       dpd_threshold: int = 90) -> pd.Series:
    """Convert a performance field into the modelling target used by real PD models.

    `y = 1` if the *booked* loan hit `dpd_threshold`+ days past due within the
    performance window. This replaces the historical-decision proxy with realised
    risk. Only defined on booked (APPROVED + disbursed) accounts.
    """
    dpd = pd.to_numeric(df[dpd_col], errors="coerce")
    return (dpd >= dpd_threshold).astype("Int64")


def retrain_with_default_label(model, booked: pd.DataFrame, rejected: pd.DataFrame | None = None,
                               dpd_col: str = "max_dpd_12m", dpd_threshold: int = 90,
                               reject_inference: str = "fuzzy"):
    """Retrain on realised default with optional reject inference.

    Parameters
    ----------
    booked : rows for disbursed loans, incl. `dpd_col` (performance).
    rejected : historically declined applications (no performance) — used for
        reject inference so the model isn't biased by the accept-only sample.
    reject_inference : 'none' | 'fuzzy'
        'fuzzy' = parcelling/fuzzy augmentation: score rejects with a booked-only
        model, assign each a fractional good/bad weight from that score, and add
        them to training. (Hard-cutoff augmentation and reweighting are alternatives.)

    Returns a fitted pipeline. This is the documented production hook; wire real
    performance data into `booked`/`rejected` to activate it.
    """
    y = make_default_label(booked, dpd_col, dpd_threshold)
    X = booked.drop(columns=[c for c in (dpd_col, C.TARGET) if c in booked.columns])
    mask = y.notna()
    X, y = X.loc[mask].reset_index(drop=True), y.loc[mask].astype(int).reset_index(drop=True)

    if reject_inference == "none" or rejected is None or len(rejected) == 0:
        pipe = build_full_pipeline(model)
        pipe.fit(X, y)
        return pipe

    # --- fuzzy augmentation ------------------------------------------------ #
    booked_only = build_full_pipeline(_clone(model)).fit(X, y)
    Xr = rejected.drop(columns=[c for c in (dpd_col, C.TARGET) if c in rejected.columns])
    p_bad = booked_only.predict_proba(Xr)[:, 1]
    # each reject enters twice: as 'bad' with weight p, as 'good' with weight 1-p
    X_aug = pd.concat([X, Xr, Xr], ignore_index=True)
    y_aug = pd.concat([y, pd.Series(np.ones(len(Xr), dtype=int)),
                       pd.Series(np.zeros(len(Xr), dtype=int))], ignore_index=True)
    w = np.concatenate([np.ones(len(X)), p_bad, 1 - p_bad])
    pipe = build_full_pipeline(model)
    pipe.fit(X_aug, y_aug, model__sample_weight=w)
    return pipe


def _clone(model):
    from sklearn.base import clone
    try:
        return clone(model)
    except Exception:
        return model


if __name__ == "__main__":
    print("Bucket-B scaffolding ready.")
    print("bureau slot ->", enable_bureau_score())
    print(bureau_expected_lift_note())
