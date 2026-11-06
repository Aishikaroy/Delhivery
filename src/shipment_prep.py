"""Shared cleaning and feature engineering for the Delay Intelligence pipeline."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def clean_shipments(raw, medians=None):
    """Clean raw shipments DataFrame. Returns (clean_df, medians_dict)."""
    df = raw.copy()

    # 1. drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. service_type: strip + title case
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()

    # 3. payment_mode: fill missing with "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. declared_value: remove rupee + commas → declared_value_inr
    if "declared_value" in df.columns:
        s = df["declared_value"].astype(str)
        s = s.str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False)
        s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
        df["declared_value_inr"] = pd.to_numeric(s, errors="coerce")
        df = df.drop(columns=["declared_value"])

    # 5. dispatch_time → datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], errors="coerce")

    # 6. parcel_weight_kg: values > 100 are sentinel → missing
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. fill missing with medians
    cols_to_fill = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
    if medians is None:
        medians = {c: float(df[c].median()) for c in cols_to_fill}
    for c in cols_to_fill:
        df[c] = df[c].fillna(medians[c])

    return df, medians


def build_features(clean, hubs):
    """Attach hub info and engineer the 17 FEATURES. Works without delivery_hours."""
    df = clean.merge(hubs, on="hub_id", how="left")

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"].clip(lower=1e-6))

    return df
