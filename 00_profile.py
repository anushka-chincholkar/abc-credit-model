"""
Step 2 + Step 4 : Dataset inspection & data-quality assessment.

This is a *read-only* profiling pass. It writes a plain-text report to
reports/profile.txt and prints the highlights. Every downstream cleaning and
feature-engineering decision is justified by something surfaced here.
"""
from __future__ import annotations

import sys
import io
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import config as C  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

OUT = io.StringIO()


def p(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    OUT.write(line + "\n")


def section(title):
    p("\n" + "=" * 78)
    p(title)
    p("=" * 78)


# --------------------------------------------------------------------------- #
# Load  (read everything as string first so we can detect type problems
# ourselves rather than letting pandas silently coerce/limit)
# --------------------------------------------------------------------------- #
raw = pd.read_csv(C.TRAINING_CSV, dtype=str, keep_default_na=False, na_values=[""])
section("STEP 2 - SHAPE / HEAD / TAIL")
p("Shape:", raw.shape)
p("\nColumns:", list(raw.columns))
p("\nHEAD:\n", raw.head().to_string())
p("\nTAIL:\n", raw.tail().to_string())

# Typed load for numeric analysis
df = pd.read_csv(C.TRAINING_CSV)
section("STEP 2 - DTYPES / INFO")
buf = io.StringIO()
df.info(buf=buf)
p(buf.getvalue())

section("STEP 2 - DESCRIBE (numeric)")
p(df.describe().to_string())
section("STEP 2 - DESCRIBE (object)")
p(df.describe(include="object").to_string())

# --------------------------------------------------------------------------- #
# Column-role census
# --------------------------------------------------------------------------- #
section("STEP 2 - COLUMN ROLE CENSUS (cardinality + missing)")
rows = []
for col in df.columns:
    nun = df[col].nunique(dropna=True)
    miss = df[col].isna().sum()
    rows.append(
        dict(
            column=col,
            dtype=str(df[col].dtype),
            n_unique=nun,
            missing=miss,
            missing_pct=round(100 * miss / len(df), 3),
            example=repr(df[col].dropna().iloc[0]) if df[col].notna().any() else "NA",
        )
    )
census = pd.DataFrame(rows)
p(census.to_string(index=False))

# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
section("STEP 4 - TARGET DISTRIBUTION / CLASS BALANCE")
vc = df[C.TARGET].value_counts(dropna=False)
p(vc.to_string())
p("\nProportions:")
p((df[C.TARGET].value_counts(normalize=True) * 100).round(3).to_string())
if len(vc) == 2:
    ratio = vc.max() / vc.min()
    p(f"\nImbalance ratio (majority/minority): {ratio:.2f} : 1")

# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #
section("STEP 4 - DUPLICATES")
p("Full-row duplicates:", df.duplicated().sum())
feat_cols = [c for c in df.columns if c != C.ID_COL]
p("Duplicates ignoring ID:", df.duplicated(subset=feat_cols).sum())
dup_feat_diff_target = (
    df.duplicated(subset=[c for c in feat_cols if c != C.TARGET], keep=False).sum()
)
p("Rows sharing identical features (incl. possible target conflicts):", dup_feat_diff_target)
# Contradictory labels: identical predictors, different Decision
pred_cols = [c for c in feat_cols if c != C.TARGET]
grp = df.groupby(pred_cols, dropna=False)[C.TARGET].nunique()
p("Predictor-identical groups with >1 distinct Decision (label noise):", int((grp > 1).sum()))

# --------------------------------------------------------------------------- #
# Numeric validity / outliers
# --------------------------------------------------------------------------- #
section("STEP 4 - NUMERIC VALIDITY & OUTLIERS")
for col in C.NUMERIC_COLS:
    s = pd.to_numeric(df[col], errors="coerce")
    p(f"\n-- {col} --")
    p(f"  min={s.min()}  max={s.max()}  mean={s.mean():.2f}  median={s.median():.2f}  std={s.std():.2f}")
    p(f"  non-numeric/NaN after coercion: {s.isna().sum()}")
    p(f"  zeros: {(s == 0).sum()}   negatives: {(s < 0).sum()}")
    q = s.quantile([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
    p("  quantiles:\n" + q.round(2).to_string())

age = pd.to_numeric(df["Age"], errors="coerce")
# surface the offending non-numeric Age token(s)
bad_age = df.loc[age.isna() & df["Age"].notna(), "Age"].unique()
p("Non-numeric Age tokens:", list(bad_age)[:10])
p("\nAge outside policy [18,60]:", int(((age < C.AGE_MIN) | (age > C.AGE_MAX)).sum()))
p("Age < 18:", int((age < C.AGE_MIN).sum()), " Age > 60:", int((age > C.AGE_MAX).sum()))
ltv = pd.to_numeric(df["LTV"], errors="coerce")
p("LTV <= 0:", int((ltv <= 0).sum()), " LTV > 100:", int((ltv > 100).sum()), " LTV in (100,200]:", int(((ltv > 100) & (ltv <= 200)).sum()))
sal = pd.to_numeric(df["Net_salary"], errors="coerce")
p("Net_salary == 0:", int((sal == 0).sum()), " < 0:", int((sal < 0).sum()))
p("Loan_Amount <= 0:", int((pd.to_numeric(df['Loan_Amount'], errors='coerce') <= 0).sum()))

# Net_salary == 0 breakdown by employment (are these students / non-earning?)
section("STEP 4 - Net_salary == 0  BY Employment_Type")
zero_sal = df[sal == 0]
if len(zero_sal):
    p(zero_sal["Employment_Type"].value_counts().to_string())
    p("\nDecision split among salary==0 rows:")
    p(zero_sal[C.TARGET].value_counts(normalize=True).round(3).to_string())

# --------------------------------------------------------------------------- #
# Categorical hygiene: whitespace, casing, rare cats, encoding artefacts
# --------------------------------------------------------------------------- #
section("STEP 4 - CATEGORICAL HYGIENE")
for col in C.CATEGORICAL_COLS + C.ORDINAL_COLS + C.BINARY_COLS:
    s = df[col].astype(str)
    n_unique = s.nunique()
    has_ws = (s != s.str.strip()).sum()
    has_case_dupes = s.str.strip().str.upper().nunique() != n_unique
    p(f"\n-- {col} -- unique={n_unique}  leading/trailing-ws={has_ws}  case-collapsible={has_case_dupes}")
    if n_unique <= 25:
        p(s.value_counts(dropna=False).to_string())
    else:
        p("  (high-ish cardinality; top 15)")
        p(s.value_counts(dropna=False).head(15).to_string())

# Non-ASCII / mojibake in text + make/model fields
section("STEP 4 - NON-ASCII / MOJIBAKE SCAN")
for col in C.TEXT_COLS + ["Make_Code", "Model_Variant"]:
    s = df[col].astype(str)
    non_ascii = s[s.str.contains(r"[^\x00-\x7F]", regex=True, na=False)]
    p(f"\n-- {col} -- rows with non-ASCII chars: {len(non_ascii)}")
    if len(non_ascii):
        p("  examples:")
        for v in non_ascii.drop_duplicates().head(6):
            p("   ", repr(v))

# --------------------------------------------------------------------------- #
# High cardinality
# --------------------------------------------------------------------------- #
section("STEP 4 - HIGH CARDINALITY FIELDS")
for col in C.HIGH_CARD_COLS + C.TEXT_COLS:
    s = df[col].astype(str)
    p(f"{col:20s} n_unique={s.nunique():6d}   top freq%={100*s.value_counts(normalize=True).iloc[0]:.2f}   singletons={int((s.value_counts()==1).sum())}")

# Rare categories (< 0.5%) per modeled categorical
section("STEP 4 - RARE CATEGORIES (<0.5% frequency)")
for col in C.CATEGORICAL_COLS + C.ORDINAL_COLS:
    vc2 = df[col].astype(str).value_counts(normalize=True)
    rare = vc2[vc2 < 0.005]
    if len(rare):
        p(f"{col}: {len(rare)} rare levels -> {list(rare.index)[:12]}")

# --------------------------------------------------------------------------- #
# Leakage / signal scan: approval rate by category + numeric AUC
# --------------------------------------------------------------------------- #
section("STEP 4/EDA - DECLINE RATE BY CATEGORICAL (signal & leakage check)")
y = (df[C.TARGET] == C.POSITIVE_LABEL).astype(int)
base_rate = y.mean()
p(f"Base DECLINED rate: {base_rate:.4f}")
for col in C.CATEGORICAL_COLS + C.ORDINAL_COLS + C.BINARY_COLS:
    g = df.assign(_y=y).groupby(col, dropna=False)["_y"].agg(["mean", "count"])
    g = g.sort_values("mean", ascending=False)
    # flag near-perfect separation (potential leakage)
    perfect = ((g["mean"] < 0.01) | (g["mean"] > 0.99)).sum()
    p(f"\n-- {col} -- (levels with ~0/100% decline rate: {perfect})")
    p(g.head(12).round(4).to_string())

section("STEP 4 - NUMERIC UNIVARIATE SEPARATION (AUC vs DECLINED)")
from sklearn.metrics import roc_auc_score  # noqa: E402
for col in C.NUMERIC_COLS:
    s = pd.to_numeric(df[col], errors="coerce")
    mask = s.notna()
    try:
        auc = roc_auc_score(y[mask], s[mask])
        p(f"{col:14s} univariate AUC={auc:.4f}  (0.5 = no signal; near 0/1 = strong / possible leak)")
    except Exception as e:
        p(f"{col}: {e}")

# Correlation among numerics
section("EDA - NUMERIC CORRELATION")
num = df[C.NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
p(num.corr().round(3).to_string())

# --------------------------------------------------------------------------- #
with open(C.REPORT_DIR / "profile.txt", "w") as f:
    f.write(OUT.getvalue())
p("\n[written to reports/profile.txt]")
