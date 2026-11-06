"""Part A — Data Foundation and Delay Evidence."""
import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES, PROMISED_HOURS

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# --- cleaning diagnostics (before filling) ---
raw_rows = len(raw)
df_dedup = raw.drop_duplicates().reset_index(drop=True)
duplicate_rows_removed = raw_rows - len(df_dedup)
rows_after_dedup = len(df_dedup)

# missing counts after dedup, before any other cleaning
missing_distance_km_before = int(df_dedup["distance_km"].isna().sum())
# sentinel weights are still present; exclude them from missing_parcel_weight_before
sentinel_mask = df_dedup["parcel_weight_kg"] > 100
sentinel_weight_rows = int(sentinel_mask.sum())
missing_parcel_weight_before = int(df_dedup["parcel_weight_kg"].isna().sum())  # does not include sentinel
missing_declared_value_before = int(df_dedup["declared_value"].isna().sum())
missing_payment_mode_before = int(df_dedup["payment_mode"].isna().sum())

service_type_spellings_before = int(df_dedup["service_type"].nunique())

# now run full clean
clean, medians = clean_shipments(raw, medians=None)
service_type_spellings_after = int(clean["service_type"].nunique())

# build features
feat = build_features(clean, hubs)
feat["is_delayed"] = (feat["delivery_hours"] > feat["promised_hours"]).astype(int)

# ensure no nulls in features + target
nulls_in_features_and_target = int(feat[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())
rows_after_hub_merge = len(feat)

# declared value stats on filled column
declared_value_mean_inr = float(feat["declared_value_inr"].mean())
declared_value_median_inr = float(feat["declared_value_inr"].median())
log_declared_value_mean = float(feat["log_declared_value"].mean())
log_declared_value_median = float(feat["log_declared_value"].median())

delayed_count = int(feat["is_delayed"].sum())
delay_rate = float(feat["is_delayed"].mean())

# hours quartiles & outliers (1.5 * IQR)
q1 = float(feat["delivery_hours"].quantile(0.25))
q3 = float(feat["delivery_hours"].quantile(0.75))
iqr = q3 - q1
hours_lower_bound = q1 - 1.5 * iqr
hours_upper_bound = q3 + 1.5 * iqr
hours_outlier_count = int(((feat["delivery_hours"] < hours_lower_bound) | (feat["delivery_hours"] > hours_upper_bound)).sum())

# write modelling table
feat.to_csv("data/modelling_table.csv", index=False)

# checks file
checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows_removed),
    ("rows_after_dedup", rows_after_dedup),
    ("missing_distance_km_before", missing_distance_km_before),
    ("missing_parcel_weight_before", missing_parcel_weight_before),
    ("sentinel_weight_rows", sentinel_weight_rows),
    ("missing_declared_value_before", missing_declared_value_before),
    ("missing_payment_mode_before", missing_payment_mode_before),
    ("service_type_spellings_before", service_type_spellings_before),
    ("service_type_spellings_after", service_type_spellings_after),
    ("median_distance_km", round(medians["distance_km"], 2)),
    ("median_parcel_weight_kg", round(medians["parcel_weight_kg"], 2)),
    ("median_declared_value_inr", round(medians["declared_value_inr"], 1)),
    ("rows_after_hub_merge", rows_after_hub_merge),
    ("nulls_in_features_and_target", nulls_in_features_and_target),
    ("delayed_count", delayed_count),
    ("delay_rate", round(delay_rate, 4)),
    ("hours_q1", round(q1, 2)),
    ("hours_q3", round(q3, 2)),
    ("hours_lower_bound", round(hours_lower_bound, 2)),
    ("hours_upper_bound", round(hours_upper_bound, 2)),
    ("hours_outlier_count", hours_outlier_count),
    ("declared_value_mean_inr", round(declared_value_mean_inr, 2)),
    ("declared_value_median_inr", round(declared_value_median_inr, 1)),
    ("log_declared_value_mean", round(log_declared_value_mean, 3)),
    ("log_declared_value_median", round(log_declared_value_median, 3)),
]
pd.DataFrame(checks, columns=["metric", "value"]).to_csv("outputs/part_a_checks.csv", index=False)

# delay by group
def delay_group(df, dim, groups=None):
    rows = []
    if dim == "distance_band":
        bins = [0, 200, 400, 600, 2000]
        labels = ["0-200", "201-400", "401-600", "601+"]
        df = df.copy()
        df["distance_band"] = pd.cut(df["distance_km"], bins=bins, labels=labels, right=True, include_lowest=True)
        for g in labels:
            sub = df[df["distance_band"] == g]
            rows.append({
                "dimension": dim, "group": g,
                "shipments": len(sub), "delayed": int(sub["is_delayed"].sum()),
                "delay_rate": round(sub["is_delayed"].mean(), 4) if len(sub) else 0.0,
                "mean_hours": round(sub["delivery_hours"].mean(), 4) if len(sub) else 0.0,
            })
    else:
        for g, sub in df.groupby(dim):
            rows.append({
                "dimension": dim, "group": str(g),
                "shipments": len(sub), "delayed": int(sub["is_delayed"].sum()),
                "delay_rate": round(sub["is_delayed"].mean(), 4),
                "mean_hours": round(sub["delivery_hours"].mean(), 4),
            })
    return rows

rows = []
rows += delay_group(feat, "service_type")
rows += delay_group(feat, "hub_tier")
rows += delay_group(feat, "distance_band")
pd.DataFrame(rows).to_csv("outputs/part_a_delay_by_group.csv", index=False)

# correlations of numeric columns
num = feat.select_dtypes(include=[np.number])
corr = num.corr().round(4)
corr.insert(0, "column", corr.index)
corr.to_csv("outputs/part_a_correlations.csv", index=False)

print("Part A complete. modelling_table rows:", len(feat))
print("delay_rate:", round(delay_rate, 4), "delayed:", delayed_count)
print("medians:", medians)
