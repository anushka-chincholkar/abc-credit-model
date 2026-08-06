"""
Central configuration for the ABC Credit loan-approval model.

Single source of truth for column roles, paths, seeds and business constants.
Keeping this here (rather than sprinkling literals through the code) makes the
pipeline reproducible and easy to audit -- a hard requirement for a credit model.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
LOG_DIR = PROJECT_ROOT / "logs"

TRAINING_CSV = DATA_DIR / "training_data.csv"

for _d in (ARTIFACT_DIR, REPORT_DIR, FIG_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Column roles (from the data dictionary in the project brief)
# --------------------------------------------------------------------------- #
TARGET = "Decision"
ID_COL = "ID"

# The positive class is DECLINED: the brief says "pay particular attention to
# recall on the DECLINED class, since that measures how many risky applicants
# you actually catch." Framing DECLINED as positive makes recall/precision read
# directly as "risky applicants caught / flagged".
POSITIVE_LABEL = "DECLINED"
NEGATIVE_LABEL = "APPROVED"

# Raw numeric predictors
NUMERIC_COLS = ["Age", "Loan_Amount", "LTV", "Net_salary"]

# Low/medium-cardinality categoricals -> one-hot (unseen-safe: handle_unknown="ignore")
# Cardinalities observed in training: Qualifications 6, Employment_Type 8, Gender 2,
# Product_Code 4, Resident_Type 3, Make_Code 12, Model_Variant 21 -> all OHE-friendly.
CATEGORICAL_COLS = [
    "Qualifications",
    "Employment_Type",
    "Gender",
    "Product_Code",
    "Resident_Type",
    "Make_Code",
    "Model_Variant",
]

# Ordinal categorical (natural order encoded in the numeric prefix, large city -> rural)
ORDINAL_COLS = ["Final_Tier"]
FINAL_TIER_ORDER = [
    "01 Megapolis (A)",
    "02 Megapolis (B)",
    "03 Large Metro",
    "04 Metro",
    "05 Urban",
    "06 Semi-Urban",
    "07 Rural",
]

# Binary flag
BINARY_COLS = ["PAST_LOANS_ACTIVE"]

# High-cardinality categoricals -> target encoding via sklearn TargetEncoder,
# which cross-fits internally (leakage-safe). Pincode 13,999 levels;
# Model_Description 87 levels (kept as high-card rather than OHE to stay compact
# and to lean on smoothed target stats).
HIGH_CARD_COLS = ["Pincode", "Model_Description"]

# Free-text (used only to derive engineered features such as engine displacement)
TEXT_COLS = ["Model_Description"]

# Business policy constants (from brief: "Policy applies to ages 18 to 60")
AGE_MIN, AGE_MAX = 18, 60
LTV_MIN, LTV_MAX = 0.0, 100.0  # a loan-to-value ratio above 100% is implausible

# CV / split configuration
N_SPLITS = 5
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # of the remaining after test is carved out
