"""
Steps 13-19 : Final evaluation, threshold selection, imbalance study, error
analysis, robustness, model selection & artifact persistence.

Flow (leakage-safe):
  tune params (read)  ->  build final estimator
  imbalance study on TRAIN (3-fold CV, resampling INSIDE folds)
  fit on TRAIN  ->  choose threshold on VALIDATION (business cost rule)
  refit on TRAIN+VAL  ->  report all metrics on held-out TEST
  error analysis on TEST  ->  save artifacts + test predictions
"""
from __future__ import annotations
import sys, json, time, warnings
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

import config as C
from data import make_splits
from pipeline import build_full_pipeline, build_preprocessor
from features import FeatureEngineer
from modeling import full_metric_block, SCALE_POS_WEIGHT

warnings.filterwarnings("ignore")
t0 = time.time()

# Business cost ratio: cost(false approval) / cost(false rejection).
# A default ~ loses loan principal+collection; a wrongful decline ~ loses the
# interest margin on one good loan. 8:1 is our defensible base case (sensitivity
# to 4:1 and 12:1 is reported).
COST_FA_OVER_FR = 8.0


# --------------------------------------------------------------------------- #
def build_tuned_estimator():
    """Instantiate the best-tuned family from reports/tuning_results.json."""
    with open(C.REPORT_DIR / "tuning_results.json") as f:
        tuning = json.load(f)
    fam = max(tuning, key=lambda k: tuning[k]["best_pr_auc_cv3"])
    params = tuning[fam]["best_params"]
    seed = C.RANDOM_SEED
    if fam == "LightGBM":
        from lightgbm import LGBMClassifier
        est = LGBMClassifier(class_weight="balanced", n_jobs=-1, random_state=seed,
                             verbose=-1, **params)
    elif fam == "XGBoost":
        from xgboost import XGBClassifier
        est = XGBClassifier(tree_method="hist", eval_metric="aucpr", n_jobs=-1,
                            random_state=seed, **params)
    elif fam == "CatBoost":
        from catboost import CatBoostClassifier
        est = CatBoostClassifier(auto_class_weights="Balanced", random_seed=seed,
                                 verbose=0, allow_writing_files=False, **params)
    else:  # HistGradientBoosting
        from sklearn.ensemble import HistGradientBoostingClassifier
        est = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **params)
    return fam, est, params


def resampled_pipeline(sampler, est):
    """imblearn pipeline: FE -> preprocess -> sampler -> estimator (sampler runs
    only on the training fold, never on validation -> leakage-safe)."""
    from imblearn.pipeline import Pipeline as ImbPipeline
    return ImbPipeline([
        ("fe", FeatureEngineer()),
        ("pre", build_preprocessor()),
        ("sampler", sampler),
        ("model", est),
    ])


def main():
    s = make_splits()
    X_train, y_train = s["train"]
    X_val, y_val = s["val"]
    X_test, y_test = s["test"]

    fam, est, params = build_tuned_estimator()
    print(f"[{time.time()-t0:.1f}s] Best family = {fam}\n  params={params}", flush=True)

    # ===================================================================== #
    # STEP 17 : imbalance-handling study (3-fold CV PR-AUC on TRAIN)
    # ===================================================================== #
    print("\n=== STEP 17: imbalance strategy comparison (CV PR-AUC / recall) ===", flush=True)
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.ensemble import BalancedRandomForestClassifier
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=C.RANDOM_SEED)

    def cv_prauc(pipe):
        return float(np.mean(cross_val_score(pipe, X_train, y_train, cv=cv3,
                                              scoring="average_precision", n_jobs=1)))
    def cv_recall(pipe):
        return float(np.mean(cross_val_score(pipe, X_train, y_train, cv=cv3,
                                              scoring="recall", n_jobs=1)))

    imb = {}
    # (a) class-weight (our tuned default)
    imb["class_weight (tuned)"] = (cv_prauc(build_full_pipeline(est)),
                                   cv_recall(build_full_pipeline(est)))
    # (b) SMOTE, (c) ADASYN -- estimator WITHOUT class weight to isolate resampling
    est_noweight = _strip_weight(fam, params)
    imb["SMOTE"] = (cv_prauc(resampled_pipeline(SMOTE(random_state=C.RANDOM_SEED), est_noweight)),
                    cv_recall(resampled_pipeline(SMOTE(random_state=C.RANDOM_SEED), est_noweight)))
    imb["ADASYN"] = (cv_prauc(resampled_pipeline(ADASYN(random_state=C.RANDOM_SEED), est_noweight)),
                     cv_recall(resampled_pipeline(ADASYN(random_state=C.RANDOM_SEED), est_noweight)))
    # (d) BalancedRandomForest
    brf = build_full_pipeline(BalancedRandomForestClassifier(
        n_estimators=300, random_state=C.RANDOM_SEED, n_jobs=-1,
        sampling_strategy="all", replacement=True, bootstrap=True))
    imb["BalancedRandomForest"] = (cv_prauc(brf), cv_recall(brf))

    imb_tbl = pd.DataFrame({k: {"pr_auc": v[0], "recall@0.5": v[1]} for k, v in imb.items()}).T
    imb_tbl = imb_tbl.sort_values("pr_auc", ascending=False)
    print(imb_tbl.round(4).to_string(), flush=True)
    imb_tbl.to_csv(C.REPORT_DIR / "imbalance_study.csv")

    # ===================================================================== #
    # Fit final estimator on TRAIN, choose threshold on VALIDATION
    # ===================================================================== #
    print(f"\n[{time.time()-t0:.1f}s] Fitting {fam} on TRAIN for threshold selection...", flush=True)
    pipe = build_full_pipeline(est)
    pipe.fit(X_train, y_train)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    print(f"  VAL  ROC-AUC={roc_auc_score(y_val, val_proba):.4f}  PR-AUC={average_precision_score(y_val, val_proba):.4f}", flush=True)

    thr = choose_thresholds(y_val.values, val_proba)
    print("\n=== STEP 14: candidate operating points (chosen on VALIDATION) ===", flush=True)
    for name, d in thr.items():
        print(f"  {name:22s} thr={d['threshold']:.3f}  recall={d['recall']:.3f}  "
              f"precision={d['precision']:.3f}  f1={d['f1']:.3f}", flush=True)
    chosen = thr["cost_min(8:1)"]["threshold"]
    print(f"\nChosen threshold = {chosen:.4f}  (cost-min at FA:FR = {COST_FA_OVER_FR:.0f}:1)", flush=True)
    plot_threshold_curves(y_val.values, val_proba, chosen)

    # ===================================================================== #
    # STEP 18/19 : refit on TRAIN+VAL, evaluate on TEST, save artifacts
    # ===================================================================== #
    X_tv = pd.concat([X_train, X_val], ignore_index=True)
    y_tv = pd.concat([y_train, y_val], ignore_index=True)
    print(f"\n[{time.time()-t0:.1f}s] Refitting on TRAIN+VAL (n={len(X_tv):,}) for deployment...", flush=True)
    final_pipe = build_full_pipeline(est)
    final_pipe.fit(X_tv, y_tv)

    test_proba = final_pipe.predict_proba(X_test)[:, 1]
    metrics_chosen = full_metric_block(y_test.values, test_proba, chosen)
    metrics_half = full_metric_block(y_test.values, test_proba, 0.5)
    print("\n=== STEP 13: FINAL TEST METRICS (chosen threshold) ===", flush=True)
    for k, v in metrics_chosen.items():
        print(f"  {k:20s}: {v}", flush=True)

    # ---- error analysis (Step 15) ---------------------------------------- #
    err = error_analysis(X_test, y_test.values, test_proba, chosen)

    # ---- robustness: seed stability (Step 17) ---------------------------- #
    seed_scores = seed_stability(X_train, y_train, X_val, y_val, fam, params)
    print(f"\nSeed stability VAL PR-AUC: mean={np.mean(seed_scores):.4f} "
          f"std={np.std(seed_scores):.4f}  {np.round(seed_scores,4).tolist()}", flush=True)

    # ---- persist artifacts ----------------------------------------------- #
    feat_names = list(final_pipe.named_steps["pre"].get_feature_names_out())
    joblib.dump(final_pipe, C.ARTIFACT_DIR / "model_pipeline.joblib")
    meta = {
        "family": fam, "hyperparameters": params,
        "threshold": chosen, "cost_fa_over_fr": COST_FA_OVER_FR,
        "positive_label": C.POSITIVE_LABEL, "negative_label": C.NEGATIVE_LABEL,
        "label_mapping": {"APPROVED": 0, "DECLINED": 1},
        "n_features": len(feat_names), "feature_names": feat_names,
        "test_metrics_chosen": metrics_chosen, "test_metrics_0.5": metrics_half,
        "val_operating_points": thr,
        "imbalance_study": imb_tbl.round(5).to_dict(orient="index"),
        "seed_stability_val_prauc": {"mean": float(np.mean(seed_scores)),
                                     "std": float(np.std(seed_scores))},
        "trained_on": "train+val", "n_train_rows": int(len(X_tv)),
        "random_seed": C.RANDOM_SEED,
    }
    with open(C.ARTIFACT_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    # test predictions in submission-style format
    sub = pd.DataFrame({
        "prob_DECLINED": test_proba.round(6),
        "prob_APPROVED": (1 - test_proba).round(6),
        "Decision": np.where(test_proba >= chosen, "DECLINED", "APPROVED"),
        "actual": np.where(y_test.values == 1, "DECLINED", "APPROVED"),
    })
    sub.to_csv(C.REPORT_DIR / "test_predictions.csv", index=False)

    with open(C.REPORT_DIR / "final_metrics.json", "w") as f:
        json.dump({"family": fam, "threshold": chosen,
                   "test_chosen": metrics_chosen, "test_0.5": metrics_half,
                   "error_analysis": err}, f, indent=2, default=float)
    print(f"\n[{time.time()-t0:.1f}s] DONE. Artifacts -> artifacts/  Reports -> reports/", flush=True)


# --------------------------------------------------------------------------- #
def _strip_weight(fam, params):
    """Same tuned params but without built-in class weighting (for SMOTE/ADASYN)."""
    seed = C.RANDOM_SEED
    p = {k: v for k, v in params.items() if k != "scale_pos_weight"}
    if fam == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_jobs=-1, random_state=seed, verbose=-1, **p)
    if fam == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=seed, **p)
    if fam == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=seed, verbose=0, allow_writing_files=False, **p)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=seed, **p)


def choose_thresholds(y_true, proba):
    """Return several candidate operating points computed on validation."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]  # align with thr
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0)
    beta2 = 5 * prec * rec / (4 * prec + rec + 1e-12)  # F-beta, beta=2 (recall-weighted)

    out = {}
    out["max_F1"] = _pack(thr[np.argmax(f1)], y_true, proba)
    out["max_F2"] = _pack(thr[np.argmax(beta2)], y_true, proba)
    # recall >= 0.70 with best precision
    mask = rec >= 0.70
    if mask.any():
        idx = np.where(mask)[0][np.argmax(prec[mask])]
        out["recall>=0.70"] = _pack(thr[idx], y_true, proba)
    # cost-minimising thresholds for several FA:FR ratios
    for R in (4.0, COST_FA_OVER_FR, 12.0):
        out[f"cost_min({R:.0f}:1)"] = _cost_min(y_true, proba, R)
    return out


def _pack(t, y_true, proba):
    from sklearn.metrics import precision_score, recall_score, f1_score
    yp = (proba >= t).astype(int)
    return {"threshold": float(t),
            "precision": float(precision_score(y_true, yp, zero_division=0)),
            "recall": float(recall_score(y_true, yp, zero_division=0)),
            "f1": float(f1_score(y_true, yp, zero_division=0))}


def _cost_min(y_true, proba, R):
    grid = np.linspace(0.01, 0.99, 197)
    best_t, best_c = 0.5, np.inf
    for t in grid:
        yp = (proba >= t).astype(int)
        fn = int(((y_true == 1) & (yp == 0)).sum())   # false approval (costly)
        fp = int(((y_true == 0) & (yp == 1)).sum())   # false rejection
        cost = R * fn + fp
        if cost < best_c:
            best_c, best_t = cost, t
    d = _pack(best_t, y_true, proba)
    d["exp_cost"] = float(best_c)
    return d


def plot_threshold_curves(y_true, proba, chosen):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    ax[0].plot(rec, prec, color="#2E86AB"); ax[0].set_xlabel("Recall (DECLINED)")
    ax[0].set_ylabel("Precision"); ax[0].set_title(f"PR curve (AP={average_precision_score(y_true, proba):.3f})")
    ax[1].plot(fpr, tpr, color="#6A4C93"); ax[1].plot([0, 1], [0, 1], "k--", alpha=.4)
    ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].set_title(f"ROC (AUC={roc_auc_score(y_true, proba):.3f})")
    t = thr
    p2, r2 = prec[:-1], rec[:-1]
    f1 = np.where((p2 + r2) > 0, 2 * p2 * r2 / (p2 + r2 + 1e-12), 0)
    ax[2].plot(t, p2, label="precision", color="#2E86AB")
    ax[2].plot(t, r2, label="recall", color="#E4572E")
    ax[2].plot(t, f1, label="F1", color="#3B8C3B")
    ax[2].axvline(chosen, color="k", ls="--", label=f"chosen={chosen:.2f}")
    ax[2].set_xlabel("threshold"); ax[2].set_title("Precision / Recall / F1 vs threshold"); ax[2].legend()
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "07_threshold_curves.png", dpi=120); plt.close()
    print("  -> reports/figures/07_threshold_curves.png", flush=True)


def error_analysis(X_test, y_true, proba, thr):
    from features import FeatureEngineer
    yp = (proba >= thr).astype(int)
    dfe = FeatureEngineer().fit_transform(X_test).copy()
    dfe["y_true"], dfe["y_pred"], dfe["proba"] = y_true, yp, proba
    fn = dfe[(dfe.y_true == 1) & (dfe.y_pred == 0)]  # false approvals (missed risky)
    fp = dfe[(dfe.y_true == 0) & (dfe.y_pred == 1)]  # false rejections (lost good)
    out = {"n_false_approval_FN": int(len(fn)), "n_false_rejection_FP": int(len(fp))}
    print("\n=== STEP 15: ERROR ANALYSIS (TEST) ===", flush=True)
    print(f"  False approvals (missed risky, FN): {len(fn)}", flush=True)
    print(f"  False rejections (lost good, FP):   {len(fp)}", flush=True)
    for col in ["LTV", "Loan_Amount", "Net_salary", "income_to_loan"]:
        a = pd.to_numeric(dfe[col], errors="coerce").astype(float)
        out[f"FN_median_{col}"] = float(a[fn.index].median())
        out[f"FP_median_{col}"] = float(a[fp.index].median())
        print(f"  {col:16s} FN_median={out[f'FN_median_{col}']:.2f}  FP_median={out[f'FP_median_{col}']:.2f}", flush=True)
    # where do misses concentrate?
    if len(fn):
        seg = fn["ltv_band"].value_counts(normalize=True).round(3).to_dict()
        out["FN_by_ltv_band"] = seg
        print("  Missed-risky (FN) by ltv_band:", seg, flush=True)
    return out


def seed_stability(X_train, y_train, X_val, y_val, fam, params):
    scores = []
    for sd in (7, 21, 101):
        p = dict(params)
        if fam == "LightGBM":
            from lightgbm import LGBMClassifier
            e = LGBMClassifier(class_weight="balanced", n_jobs=-1, random_state=sd, verbose=-1, **p)
        elif fam == "XGBoost":
            from xgboost import XGBClassifier
            e = XGBClassifier(tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=sd, **p)
        elif fam == "CatBoost":
            from catboost import CatBoostClassifier
            e = CatBoostClassifier(auto_class_weights="Balanced", random_seed=sd, verbose=0, allow_writing_files=False, **p)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            e = HistGradientBoostingClassifier(class_weight="balanced", random_state=sd, **p)
        pp = build_full_pipeline(e); pp.fit(X_train, y_train)
        scores.append(average_precision_score(y_val, pp.predict_proba(X_val)[:, 1]))
    return np.array(scores)


if __name__ == "__main__":
    main()
