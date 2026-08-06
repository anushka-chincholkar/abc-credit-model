"""
Step (added) : Probability calibration + calibrated-PD threshold + reason codes.

  * Fit base tuned CatBoost on TRAIN.
  * Fit IsotonicRegression PD calibrator on VALIDATION (held out from training).
  * Evaluate calibration on TEST: Brier score + Expected Calibration Error + a
    reliability diagram (fig 13), before vs after.
  * Map the chosen operating point into calibrated-PD space (predictions preserved,
    since isotonic is monotone) -> the threshold becomes an interpretable PD.
  * Save the CalibratedPipeline artifact + updated metadata; demo reason codes.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

import config as C
from data import make_splits
from pipeline import build_full_pipeline
from calibration import CalibratedPipeline
from modeling import full_metric_block

RAW_THR = 0.59  # chosen operating point in raw-score space (from 07)


def ece(y, p, bins=10):
    """Expected Calibration Error (equal-width bins)."""
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return e


def main():
    s = make_splits()
    X_tr, y_tr = s["train"]; X_val, y_val = s["val"]; X_te, y_te = s["test"]

    # tuned CatBoost
    with open(C.REPORT_DIR / "tuning_results.json") as f:
        params = json.load(f)["CatBoost"]["best_params"]
    from catboost import CatBoostClassifier
    est = CatBoostClassifier(auto_class_weights="Balanced", random_seed=C.RANDOM_SEED,
                             verbose=0, allow_writing_files=False, **params)
    base = build_full_pipeline(est)
    print("Fitting base on TRAIN...", flush=True)
    base.fit(X_tr, y_tr)

    raw_val = base.predict_proba(X_val)[:, 1]
    raw_te = base.predict_proba(X_te)[:, 1]

    # isotonic PD calibrator fit on VALIDATION
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_val, y_val.values)
    feat_names = list(base.named_steps["pre"].get_feature_names_out())
    cal = CalibratedPipeline(base, iso, feat_names)

    pd_te = iso.transform(raw_te)
    y = y_te.values

    # ranking preserved?
    print("\n=== ranking preserved by isotonic (monotone) ===", flush=True)
    print(f"  ROC-AUC raw={roc_auc_score(y, raw_te):.4f}  calibrated={roc_auc_score(y, pd_te):.4f}", flush=True)
    print(f"  PR-AUC  raw={average_precision_score(y, raw_te):.4f}  calibrated={average_precision_score(y, pd_te):.4f}", flush=True)

    # calibration quality on TEST
    b_raw, b_cal = brier_score_loss(y, raw_te), brier_score_loss(y, pd_te)
    e_raw, e_cal = ece(y, raw_te), ece(y, pd_te)
    print("\n=== calibration quality on TEST (lower = better) ===", flush=True)
    print(f"  Brier: raw={b_raw:.4f}  calibrated={b_cal:.4f}", flush=True)
    print(f"  ECE:   raw={e_raw:.4f}  calibrated={e_cal:.4f}", flush=True)
    print(f"  mean predicted PD calibrated={pd_te.mean():.4f}  vs actual bad rate={y.mean():.4f}", flush=True)

    # reliability diagram
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for lab, p, c in [("raw", raw_te, "#E4572E"), ("calibrated", pd_te, "#2E86AB")]:
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax[0].plot(mean, frac, "o-", label=lab, color=c)
    ax[0].plot([0, 1], [0, 1], "k--", alpha=.4)
    ax[0].set_xlabel("mean predicted PD"); ax[0].set_ylabel("observed bad rate")
    ax[0].set_title("Reliability diagram (TEST)"); ax[0].legend()
    ax[1].hist(pd_te, bins=40, color="#2E86AB", alpha=.8)
    ax[1].axvline(float(iso.transform([RAW_THR])[0]), color="k", ls="--",
                  label=f"decision PD={float(iso.transform([RAW_THR])[0]):.3f}")
    ax[1].set_xlabel("calibrated PD"); ax[1].set_title("Calibrated PD distribution (TEST)"); ax[1].legend()
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "13_calibration.png", dpi=120); plt.close()
    print("  -> reports/figures/13_calibration.png", flush=True)

    # map operating threshold into calibrated-PD space (predictions preserved)
    cal_thr = float(iso.transform([RAW_THR])[0])
    m_raw = full_metric_block(y, raw_te, RAW_THR)
    m_cal = full_metric_block(y, pd_te, cal_thr)
    print(f"\n=== threshold mapping ===", flush=True)
    print(f"  raw threshold {RAW_THR} -> calibrated PD threshold {cal_thr:.4f}", flush=True)
    print(f"  predictions identical? recall raw={m_raw['recall_declined']:.4f} cal={m_cal['recall_declined']:.4f} "
          f"| decisions equal={np.array_equal((raw_te>=RAW_THR),(pd_te>=cal_thr))}", flush=True)
    print(f"  => Decision rule: DECLINE if calibrated PD >= {cal_thr*100:.1f}%", flush=True)

    # save calibrated artifact + metadata
    joblib.dump(cal, C.ARTIFACT_DIR / "model_calibrated.joblib")
    meta = json.loads((C.ARTIFACT_DIR / "model_metadata.json").read_text())
    meta["calibration"] = {
        "method": "isotonic", "fit_on": "validation",
        "raw_threshold": RAW_THR, "calibrated_pd_threshold": cal_thr,
        "brier_raw": b_raw, "brier_calibrated": b_cal,
        "ece_raw": e_raw, "ece_calibrated": e_cal,
        "mean_pd_test": float(pd_te.mean()), "actual_bad_rate_test": float(y.mean()),
    }
    (C.ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=float))

    # demo reason codes on 3 archetypes
    print("\n=== REASON CODES (adverse-action) demo ===", flush=True)
    demos = {
        "high_risk": {"Qualifications": "SSC", "Employment_Type": "STU", "Age": 20, "Pincode": "509210",
                      "Gender": "Male", "Product_Code": "MC", "Loan_Amount": 135000, "LTV": 97.0,
                      "Resident_Type": "R", "Net_salary": 0, "Final_Tier": "07 Rural", "Make_Code": "RAIDER",
                      "Model_Description": "TVS RAIDER DISC BSVI", "Model_Variant": "RAIDER",
                      "PAST_LOANS_ACTIVE": "NO_PAST_LOANS"},
        "low_risk": {"Qualifications": "GRAD", "Employment_Type": "SAL", "Age": 38, "Pincode": "600001",
                     "Gender": "Female", "Product_Code": "EB", "Loan_Amount": 60000, "LTV": 55.0,
                     "Resident_Type": "O", "Net_salary": 60000, "Final_Tier": "03 Large Metro",
                     "Make_Code": "TVS", "Model_Description": "TVS IQUBE UG", "Model_Variant": "EBIKE",
                     "PAST_LOANS_ACTIVE": "PAST_LOANS_ACTIVE"},
    }
    for name, a in demos.items():
        Xa = pd.DataFrame([a])
        pd_a = float(cal.predict_pd(Xa)[0])
        decision = "DECLINED" if pd_a >= cal_thr else "APPROVED"
        rc = cal.reason_codes(Xa, top=4, favourable=(decision == "APPROVED"))
        print(f"\n  {name}: calibrated PD={pd_a*100:.1f}%  -> {decision}", flush=True)
        for r in rc:
            print(f"    [{r['code']}] {r['reason']}  ({r['direction']}, shap={r['contribution']:+.3f})", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
