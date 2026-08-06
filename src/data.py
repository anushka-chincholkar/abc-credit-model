"""
Step 9 : Data loading + leakage-free split strategy.

Order of operations (matters for leakage):
  1. load raw csv
  2. deterministic clean (drops null-target + exact dupes)  -- no fitting, no target use
  3. carve a stratified HOLD-OUT TEST set first, then a VALIDATION set
  4. all *fitting* (imputers, encoders, scalers, models) happens on TRAIN only,
     via the sklearn Pipeline, inside CV folds.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

import config as C
from cleaning import clean_dataframe


def load_raw() -> pd.DataFrame:
    return pd.read_csv(C.TRAINING_CSV, dtype=str, keep_default_na=False, na_values=[""])


def get_clean_Xy():
    """Return (X, y, ids) after deterministic cleaning. y=1 for DECLINED (positive)."""
    df = load_raw()
    df = clean_dataframe(df, drop_null_target=True)
    y = (df[C.TARGET].astype("string").str.upper() == C.POSITIVE_LABEL).astype(int)
    ids = df[C.ID_COL]
    X = df.drop(columns=[C.TARGET])
    return X.reset_index(drop=True), y.reset_index(drop=True), ids.reset_index(drop=True)


def make_splits(random_state: int = C.RANDOM_SEED):
    """Stratified train / validation / test.

    test = TEST_SIZE of all; validation = VAL_SIZE of the remainder; rest = train.
    Stratification on y preserves the ~5.3% DECLINED rate in every split.
    """
    X, y, ids = get_clean_Xy()

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=random_state
    )
    val_rel = C.VAL_SIZE / (1.0 - C.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=random_state
    )
    return {
        "train": (X_train.reset_index(drop=True), y_train.reset_index(drop=True)),
        "val": (X_val.reset_index(drop=True), y_val.reset_index(drop=True)),
        "test": (X_test.reset_index(drop=True), y_test.reset_index(drop=True)),
        "full": (X, y),
    }


if __name__ == "__main__":
    s = make_splits()
    for name in ("train", "val", "test"):
        Xn, yn = s[name]
        print(f"{name:5s}  n={len(Xn):7d}  DECLINED={yn.mean():.4f}")
