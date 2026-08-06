"""
Step 8 : Preprocessing pipeline.

Builds a single sklearn ColumnTransformer that handles every column type
correctly and -- crucially for this brief -- never breaks on an unseen category:

  numeric         -> median impute + (configurable) scaler
  binary flags    -> passthrough (already 0/1, no NaN by construction)
  ordinal tier    -> impute + OrdinalEncoder(order), unknown -> -1
  low-card cats   -> impute + OneHotEncoder(handle_unknown="infrequent_if_exist")
  high-card cats  -> TargetEncoder (internally cross-fitted => leakage-safe)

The FeatureEngineer runs first (inside the same pipeline) so engineered columns
exist before the ColumnTransformer selects them.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, TargetEncoder,
    StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer,
)

import config as C
from features import (
    FeatureEngineer, ENGINEERED_NUMERIC, ENGINEERED_CAT, ENGINEERED_BINARY,
)

NUMERIC = C.NUMERIC_COLS + ENGINEERED_NUMERIC
OHE_CATS = C.CATEGORICAL_COLS + ENGINEERED_CAT
BINARY = ENGINEERED_BINARY  # 0/1 ints
ORDINAL = C.ORDINAL_COLS
HIGH_CARD = C.HIGH_CARD_COLS

_SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,       # default: resilient to the heavy Net_salary tail
    "minmax": MinMaxScaler,
    "power": lambda: PowerTransformer(method="yeo-johnson"),
    "none": None,
}


def _make_scaler(name: str):
    factory = _SCALERS[name]
    return factory() if callable(factory) else factory


def build_preprocessor(numeric_scaler: str = "robust", ohe_min_freq: float = 0.005,
                       extra_numeric: list | None = None) -> ColumnTransformer:
    """Assemble the ColumnTransformer. `numeric_scaler` lets us A/B scalers;
    `extra_numeric` appends engineered interaction columns to the numeric group."""
    numeric = NUMERIC + list(extra_numeric or [])
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    scaler = _make_scaler(numeric_scaler)
    if scaler is not None:
        num_steps.append(("scale", scaler))
    numeric_pipe = Pipeline(num_steps)

    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(
            categories=[C.FINAL_TIER_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,        # unseen tier -> sentinel, pipeline never breaks
        )),
    ])

    ohe_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(
            handle_unknown="infrequent_if_exist",  # unseen level -> infrequent bucket
            min_frequency=ohe_min_freq,            # rare levels grouped for robustness
            sparse_output=False,
        )),
    ])

    # TargetEncoder cross-fits internally during fit_transform, so no target leaks
    # into the model's training features; unseen levels fall back to the prior.
    highcard_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("te", TargetEncoder(smooth="auto", cv=5)),  # internal 5-fold cross-fit
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("bin", "passthrough", BINARY),
            ("ord", ordinal_pipe, ORDINAL),
            ("ohe", ohe_pipe, OHE_CATS),
            ("hc", highcard_pipe, HIGH_CARD),
        ],
        remainder="drop",          # ID, raw Pincode text, Model_Description dropped
        verbose_feature_names_out=True,
    )
    return pre


def build_full_pipeline(model, numeric_scaler: str = "robust", fe=None,
                        extra_numeric: list | None = None) -> Pipeline:
    """FeatureEngineer -> preprocessing -> estimator. Ready for CV / fit / predict.
    Pass a custom `fe` (e.g. InteractionEngineer) + `extra_numeric` to A/B features."""
    return Pipeline([
        ("fe", fe if fe is not None else FeatureEngineer()),
        ("pre", build_preprocessor(numeric_scaler=numeric_scaler, extra_numeric=extra_numeric)),
        ("model", model),
    ])
