"""Package the final delay classifier for production use."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# load config
cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
k = int(cfg["selector_k"])
n_est = int(cfg["n_estimators"])
max_depth = None if str(cfg["max_depth"]) == "unlimited" else int(cfg["max_depth"])
min_leaf = int(cfg["min_samples_leaf"])
threshold = float(cfg["threshold"])

# rebuild stratified split
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# fit final pipeline on training rows only
pipe = Pipeline([
    ("selector", SelectKBest(f_classif, k=k)),
    ("model", RandomForestClassifier(
        n_estimators=n_est, max_depth=max_depth,
        min_samples_leaf=min_leaf, random_state=42,
    )),
])
pipe.fit(X_train, y_train)

# save artefacts
joblib.dump(pipe, "models/delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

# medians from historical clean
raw = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw, medians=None)
pd.DataFrame([
    {"column": "distance_km", "median": medians["distance_km"]},
    {"column": "parcel_weight_kg", "median": medians["parcel_weight_kg"]},
    {"column": "declared_value_inr", "median": medians["declared_value_inr"]},
]).to_csv("models/impute_values.csv", index=False)

# reload check
reloaded = joblib.load("models/delay_model.pkl")
p1 = pipe.predict_proba(X_test)[:, 1]
p2 = reloaded.predict_proba(X_test)[:, 1]
diff = float(np.abs(p1 - p2).max())

pd.DataFrame([
    ("test_rows_scored", len(X_test)),
    ("reload_max_abs_probability_diff", diff),
    ("threshold", threshold),
], columns=["metric", "value"]).to_csv("outputs/part_d_reload_check.csv", index=False)

print("package_model done. reload_diff:", diff)
