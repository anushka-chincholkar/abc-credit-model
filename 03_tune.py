"""
Step 12 : Hyperparameter optimisation with Optuna (Bayesian TPE).

We tune the top-2 gradient-boosting families from the CV comparison (read from
reports/cv_results_table.csv). Objective = mean PR-AUC over 3-fold Stratified CV
on the *training* split, through the full leakage-safe pipeline. Grid search is
deliberately avoided (the search space is large and continuous).

Writes reports/tuning_results.json with best params + value per family.
"""
from __future__ import annotations
import sys, json, time, warnings
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config as C
from data import make_splits
from pipeline import build_full_pipeline
from modeling import SCALE_POS_WEIGHT

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
t0 = time.time()

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
seed = C.RANDOM_SEED

s = make_splits()
X_train, y_train = s["train"]
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

# Speed the search: tune on a stratified subsample (final model in 04 refits on
# the full train+val). HPO only needs a stable *ranking* of configs, which a
# 45k-row stratified sample preserves for this signal level.
TUNE_N = 45000
if len(X_train) > TUNE_N:
    from sklearn.model_selection import train_test_split as _tts
    Xs, _, ys, _ = _tts(X_train, y_train, train_size=TUNE_N, stratify=y_train, random_state=seed)
    Xs = Xs.reset_index(drop=True); ys = ys.reset_index(drop=True)
else:
    Xs, ys = X_train, y_train


def _pipe_score(est) -> float:
    sc = cross_val_score(build_full_pipeline(est), Xs, ys, cv=cv,
                         scoring="average_precision", n_jobs=1)
    return float(np.mean(sc))


def space_lightgbm(trial):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 300, 1200, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        class_weight="balanced", n_jobs=-1, random_state=seed, verbose=-1)


def space_xgboost(trial):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 300, 1200, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        gamma=trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 5.0, 30.0),
        tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=seed)


def space_catboost(trial):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=trial.suggest_int("iterations", 300, 1200, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        auto_class_weights="Balanced", random_seed=seed, verbose=0,
        allow_writing_files=False)


def space_histgb(trial):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=trial.suggest_int("max_iter", 300, 1200, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        class_weight="balanced", random_state=seed)


SPACES = {"LightGBM": space_lightgbm, "XGBoost": space_xgboost,
          "CatBoost": space_catboost, "HistGradientBoosting": space_histgb}


def pick_top_families():
    try:
        tbl = pd.read_csv(C.REPORT_DIR / "cv_results_table.csv")
        ranked = [m for m in tbl.sort_values("cv_pr_auc_mean", ascending=False)["model"]
                  if m in SPACES]
        return ranked[:2] if ranked else ["LightGBM", "CatBoost"]
    except FileNotFoundError:
        return ["LightGBM", "CatBoost"]


def main():
    families = pick_top_families()
    print(f"[{time.time()-t0:.1f}s] tuning families: {families}  ({N_TRIALS} trials each)", flush=True)
    out = {}
    for fam in families:
        builder = SPACES[fam]
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))

        def objective(trial, _b=builder):
            return _pipe_score(_b(trial))

        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        out[fam] = {"best_pr_auc_cv3": study.best_value, "best_params": study.best_params,
                    "n_trials": len(study.trials)}
        print(f"[{time.time()-t0:.1f}s] {fam}: best CV3 PR-AUC={study.best_value:.4f}", flush=True)
        print(f"    params={study.best_params}", flush=True)
        with open(C.REPORT_DIR / "tuning_results.json", "w") as f:
            json.dump(out, f, indent=2)
    print(f"\nDONE tuning in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
