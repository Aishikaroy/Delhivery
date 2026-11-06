"""Score new shipments and produce an action list."""
import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

# load artefacts
raw_new = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute = pd.read_csv("models/impute_values.csv")
medians = dict(zip(impute["column"], impute["median"]))
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

# clean + features
clean, _ = clean_shipments(raw_new, medians=medians)
feat = build_features(clean, hubs)
X = feat[feature_order]

# score
probs = model.predict_proba(X)[:, 1]
flagged = (probs >= threshold).astype(int)

out = pd.DataFrame({
    "shipment_id": feat["shipment_id"].values,
    "hub_id": feat["hub_id"].values,
    "delay_probability": np.round(probs, 4),
    "flagged": flagged,
})
out = out.merge(segments, on="hub_id", how="left")

def action_row(row):
    if row["flagged"] == 0:
        return "No action"
    if row["segment"] == "Escalate":
        return "Escalate to hub manager"
    return "Proactive customer call"

out["action"] = out.apply(action_row, axis=1)
out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv("outputs/new_shipment_actions.csv", index=False)

print("predict_delay done. flagged:", int(out["flagged"].sum()),
      "escalate:", int((out["action"] == "Escalate to hub manager").sum()))
print(out["action"].value_counts().to_string())
