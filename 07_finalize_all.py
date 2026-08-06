"""
Finalisation:
  (1) adopt the business-recommended threshold (risk-averse, recall-focused),
      recompute TEST metrics, update artifacts + test_predictions consistently;
  (2) regenerate the partial-dependence figure (integer-column fix);
  (3) measure inference latency + validate predict_customer / chatbot end-to-end.
"""
from __future__ import annotations
import sys, json, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from modeling import full_metric_block

# Business decision: FA:FR = 12:1 (risk-averse NBFC stance, honours the brief's
# emphasis on catching risky applicants). Cost-min on validation -> ~0.59.
FINAL_THRESHOLD = 0.59

# --------------------------------------------------------------------------- #
# (1) Recompute TEST metrics at the adopted threshold; update artifacts.
sub = pd.read_csv(C.REPORT_DIR / "test_predictions.csv")
y = (sub["actual"] == "DECLINED").astype(int).values
p = sub["prob_DECLINED"].values
m_final = full_metric_block(y, p, FINAL_THRESHOLD)
m_half = full_metric_block(y, p, 0.5)
print("=== ADOPTED TEST METRICS @ thr =", FINAL_THRESHOLD, "===")
for k, v in m_final.items():
    print(f"  {k:20s}: {v}")

# operating-point table across the grid (for the report)
grid = [0.50, 0.55, 0.59, 0.62, 0.65, 0.69, 0.75]
op = [full_metric_block(y, p, t) for t in grid]
op_tbl = pd.DataFrame(op)[["threshold", "recall_declined", "precision_declined",
                           "f1_declined", "specificity", "balanced_accuracy", "mcc",
                           "tn", "fp", "fn", "tp"]]
op_tbl.to_csv(C.REPORT_DIR / "test_operating_points.csv", index=False)

sub["Decision"] = np.where(p >= FINAL_THRESHOLD, "DECLINED", "APPROVED")
sub.to_csv(C.REPORT_DIR / "test_predictions.csv", index=False)

meta = json.loads((C.ARTIFACT_DIR / "model_metadata.json").read_text())
meta["threshold"] = FINAL_THRESHOLD
meta["threshold_policy"] = "cost-min FA:FR=12:1 (risk-averse, recall-focused)"
meta["cost_fa_over_fr"] = 12.0
meta["test_metrics_chosen"] = m_final
meta["test_metrics_0.5"] = m_half
(C.ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=float))

fm = json.loads((C.REPORT_DIR / "final_metrics.json").read_text())
fm["threshold"] = FINAL_THRESHOLD
fm["test_chosen"] = m_final
(C.REPORT_DIR / "final_metrics.json").write_text(json.dumps(fm, indent=2, default=float))
print("\n[updated artifacts/model_metadata.json + reports/final_metrics.json + test_predictions.csv]")

# --------------------------------------------------------------------------- #
# (2) Partial dependence (fix: cast raw numeric inputs to float)
print("\n=== regenerating partial dependence ===")
from data import make_splits
from sklearn.inspection import PartialDependenceDisplay
pipe = joblib.load(C.ARTIFACT_DIR / "model_pipeline.joblib")
X_test, _ = make_splits()["test"]
samp = X_test.sample(n=min(4000, len(X_test)), random_state=C.RANDOM_SEED).copy()
for col in ["Age", "Loan_Amount", "LTV", "Net_salary"]:
    samp[col] = pd.to_numeric(samp[col], errors="coerce").astype("float64")
try:
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    PartialDependenceDisplay.from_estimator(
        pipe, samp, ["LTV", "Loan_Amount", "Net_salary", "Age"],
        ax=ax.ravel()[:4], response_method="predict_proba")
    plt.suptitle("Partial dependence on P(DECLINED)")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "12_partial_dependence.png", dpi=120); plt.close()
    print("  -> reports/figures/12_partial_dependence.png")
except Exception as e:
    print("  PDP still failed:", type(e).__name__, str(e)[:120])

# --------------------------------------------------------------------------- #
# (3) Latency + predict_customer + chatbot
from predict import predict_customer
from chatbot import run_session, mask_pii

ARCH = {
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
    "borderline": {"Qualifications": "HSC", "Employment_Type": "SEP", "Age": 29, "Pincode": "302001",
                   "Gender": "Male", "Product_Code": "SC", "Loan_Amount": 95000, "LTV": 84.0,
                   "Resident_Type": "O", "Net_salary": 28000, "Final_Tier": "05 Urban", "Make_Code": "JUPITER",
                   "Model_Description": "JUPITER 125 BSVI - DISC", "Model_Variant": "125 CC DISC",
                   "PAST_LOANS_ACTIVE": "NO_PAST_LOANS"},
}
print("\n=== ARCHETYPE PREDICTIONS (thr", FINAL_THRESHOLD, ") ===")
for name, a in ARCH.items():
    r = predict_customer(a)
    print(f"  {name:11s} -> {r['decision']:8s}  P(decline)={r['prob_decline']:.3f}  conf={r['confidence']:.2f}")
    print(f"      why: {r['risk_explanation'][0]}")

print("\n=== LATENCY (single applicant) ===")
predict_customer(ARCH["borderline"])  # warm
ts = []
for _ in range(500):
    t0 = time.perf_counter(); predict_customer(ARCH["borderline"]); ts.append((time.perf_counter()-t0)*1000)
ts = np.array(ts)
lat = {"single_mean_ms": float(ts.mean()), "single_p50_ms": float(np.percentile(ts, 50)),
       "single_p95_ms": float(np.percentile(ts, 95)), "single_p99_ms": float(np.percentile(ts, 99))}
print(f"  mean={lat['single_mean_ms']:.2f}ms  p50={lat['single_p50_ms']:.2f}ms  "
      f"p95={lat['single_p95_ms']:.2f}ms  p99={lat['single_p99_ms']:.2f}ms")
t0 = time.perf_counter(); pipe.predict_proba(X_test); dt = time.perf_counter()-t0
lat["batch_ms_per_applicant"] = float(1000*dt/len(X_test))
print(f"  batch: {1000*dt/len(X_test):.4f} ms/applicant over {len(X_test):,}")
(C.REPORT_DIR / "latency.json").write_text(json.dumps(lat, indent=2))

print("\n=== CHATBOT SESSION + PII MASKING ===")
payload = run_session(ARCH["high_risk"], interactive=False, session_id="DEMO-TEST")
print("  decision:", payload["decision"], " P(decline)=", payload["prob_decline"])
print("  pincode masked in log as:", mask_pii(ARCH["high_risk"])["Pincode"])
last = (C.LOG_DIR / "applications.jsonl").read_text().strip().splitlines()[-1]
print("  repository entry:", last[:200])
print("\nDONE.")
