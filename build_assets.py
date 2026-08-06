"""
Build chatbot lookup assets from the training data so the bot can DERIVE fields
instead of asking for them:
  * vehicle_catalog.json : Model_Description -> {Make_Code, Model_Variant, Product_Code}
                           (+ a popularity-ranked list for the picker)
  * pincode_tier.json    : Pincode -> most-common Final_Tier

This keeps the applicant-facing question count low: one vehicle pick fills four
model fields, and the pincode fills the tier.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "src")
import pandas as pd
import config as C
from data import load_raw
from cleaning import clean_dataframe

df = clean_dataframe(load_raw(), drop_null_target=True)

# --- vehicle catalog ------------------------------------------------------- #
veh_cols = ["Model_Description", "Make_Code", "Model_Variant", "Product_Code"]
vsub = df[veh_cols].dropna(subset=["Model_Description"])
catalog = {}
order = []
for desc, grp in vsub.groupby("Model_Description"):
    mode = grp.mode(dropna=True).iloc[0]
    catalog[desc] = {
        "Make_Code": mode["Make_Code"],
        "Model_Variant": mode["Model_Variant"],
        "Product_Code": mode["Product_Code"],
        "count": int(len(grp)),
    }
# popularity-ranked picker list (top models cover most applications)
order = sorted(catalog, key=lambda k: catalog[k]["count"], reverse=True)
popular = order[:40]

with open(C.ARTIFACT_DIR / "vehicle_catalog.json", "w") as f:
    json.dump({"catalog": catalog, "popular": popular}, f, indent=2, default=str)
print(f"vehicle_catalog.json: {len(catalog)} models, top-40 picker "
      f"(covers {sum(catalog[m]['count'] for m in popular)/len(vsub):.1%} of applications)")

# --- pincode -> tier ------------------------------------------------------- #
psub = df[["Pincode", "Final_Tier"]].dropna()
pin_tier = {}
for pin, grp in psub.groupby("Pincode"):
    pin_tier[str(pin)] = grp["Final_Tier"].mode().iloc[0]
with open(C.ARTIFACT_DIR / "pincode_tier.json", "w") as f:
    json.dump(pin_tier, f, default=str)
print(f"pincode_tier.json: {len(pin_tier)} pincodes mapped to a tier")

# region-digit -> most-common tier (fallback for pincodes not in the exact map)
psub = psub.assign(_region=psub["Pincode"].astype(str).str[0])
region_tier = {r: g["Final_Tier"].mode().iloc[0] for r, g in psub.groupby("_region")}
region_tier["_default"] = df["Final_Tier"].mode().iloc[0]      # global fallback
with open(C.ARTIFACT_DIR / "region_tier.json", "w") as f:
    json.dump(region_tier, f, default=str)
print("region_tier.json:", region_tier)

# also record the tier vocabulary + region-digit fallback
tiers = sorted(df["Final_Tier"].dropna().unique().tolist())
print("tiers:", tiers)
