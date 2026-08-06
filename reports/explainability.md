# Explainability — final model

## Permutation importance (TEST, scorer = average_precision)

- LTV                  Δ(PR-AUC)=+0.0939
- Pincode              Δ(PR-AUC)=+0.0346
- PAST_LOANS_ACTIVE    Δ(PR-AUC)=+0.0121
- Loan_Amount          Δ(PR-AUC)=+0.0115
- Model_Description    Δ(PR-AUC)=+0.0041
- Net_salary           Δ(PR-AUC)=+0.0020
- Age                  Δ(PR-AUC)=+0.0019
- Employment_Type      Δ(PR-AUC)=+0.0013
- Model_Variant        Δ(PR-AUC)=+0.0013
- Qualifications       Δ(PR-AUC)=+0.0009
- Gender               Δ(PR-AUC)=+0.0007
- Make_Code            Δ(PR-AUC)=+0.0005
- Product_Code         Δ(PR-AUC)=+0.0004
- Final_Tier           Δ(PR-AUC)=+0.0004
- Resident_Type        Δ(PR-AUC)=+0.0001

## SHAP (global + local)

Top-15 features by mean |SHAP|:
- LTV                    0.6682
- Pincode                0.2147
- past_loan_active       0.1050
- Age                    0.0978
- Product_Code_MC        0.0923
- Loan_Amount            0.0889
- down_payment_est       0.0865
- high_ltv               0.0698
- pincode_region_5       0.0487
- ltv_band_75-85         0.0455
- Model_Description      0.0421
- vehicle_value_est      0.0419
- pincode_region_8       0.0414
- Employment_Type_SAL    0.0409
- pincode_region_6       0.0353
  -> figures 09_shap_summary / 10_shap_bar / 11_shap_waterfall_highrisk

## Partial dependence (top numeric drivers)

PDP step skipped: ValueError: The column 7 contains integer data. Partial dependence plots are not supported for integer data: this can lead to implicit rounding with NumPy arrays or even er

## Business translation

- **LTV / high_ltv / down_payment** dominate: the thinner the borrower's equity in the vehicle, the higher the modelled default risk. This is the lever the chatbot should stress-test first.
- **Geography (Pincode target-encoding, tier, pincode_region)** contributes secondary signal — regional repayment behaviour. Encoded safely so unseen pincodes fall back to the regional/global prior.
- **PAST_LOANS_ACTIVE** lowers risk (existing repayment track record).
- **Self-reported Net_salary** is weak and non-monotone — consistent with it being unverified; the model leans on collateral (LTV) far more than on stated income, which is the prudent behaviour for an NBFC.
