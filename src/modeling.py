"""
Shared modeling utilities: the model zoo, CV scorers, and a full metric block.

Imbalance handling note
------------------------
DECLINED is only ~5.3% of rows. Every learner here is told about the imbalance
up front -- `class_weight="balanced"` for the sklearn/linear/tree models,
`scale_pos_weight` for XGBoost, `auto_class_weights="Balanced"` for CatBoost --
so the broad comparison is apples-to-apples. Resampling (SMOTE/ADASYN) is
compared separately in the imbalance experiment (Step 17).
"""
from __future__ import annotations
import warnings
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, balanced_accuracy_score, matthews_corrcoef,
    log_loss, confusion_matrix,
)

import config as C

# Imbalance ratio for boosters that take a scalar pos weight (~18:1)
SCALE_POS_WEIGHT = 18.0


def get_model_zoo() -> dict:
    """name -> estimator. All configured for the ~18:1 imbalance."""
    seed = C.RANDOM_SEED
    zoo = {
        "LogisticRegression": LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0, n_jobs=None),
        "LinearSVM_SGD": SGDClassifier(
            loss="modified_huber", class_weight="balanced", random_state=seed,
            max_iter=2000, tol=1e-3),  # SGD hinge-variant w/ predict_proba (RBF-SVM infeasible at 108k)
        "GaussianNB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=8, min_samples_leaf=50, random_state=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", max_depth=None,
            min_samples_leaf=20, n_jobs=-1, random_state=seed),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            min_samples_leaf=20, n_jobs=-1, random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, subsample=0.6, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=None, l2_regularization=1.0,
            class_weight="balanced", random_state=seed),
    }
    # Optional native GBMs (import-guarded so the script still runs if absent)
    try:
        from xgboost import XGBClassifier
        zoo["XGBoost"] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=SCALE_POS_WEIGHT, eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=seed)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"XGBoost unavailable: {e}")
    try:
        from lightgbm import LGBMClassifier
        zoo["LightGBM"] = LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8,
            colsample_bytree=0.8, class_weight="balanced", n_jobs=-1,
            random_state=seed, verbose=-1)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"LightGBM unavailable: {e}")
    try:
        from catboost import CatBoostClassifier
        zoo["CatBoost"] = CatBoostClassifier(
            iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
            auto_class_weights="Balanced", random_seed=seed, verbose=0,
            allow_writing_files=False)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"CatBoost unavailable: {e}")
    return zoo


def get_baselines() -> dict:
    seed = C.RANDOM_SEED
    return {
        "Dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "Dummy_stratified": DummyClassifier(strategy="stratified", random_state=seed),
    }


def full_metric_block(y_true, proba, threshold: float) -> dict:
    """Every Step-13 metric at a given probability threshold. Positive = DECLINED."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0        # TNR (APPROVED correctly kept)
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0        # == recall on DECLINED
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ll = log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1])
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_declined": precision_score(y_true, y_pred, zero_division=0),
        "recall_declined": recall_score(y_true, y_pred, zero_division=0),
        "f1_declined": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "log_loss": ll,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# CV scorers (threshold-independent ones rank the broad comparison)
CV_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
}
