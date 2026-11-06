"""Part D — Hub Segments."""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# hub profile
profile = df.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean"),
).reset_index().sort_values("hub_id").reset_index(drop=True)

for c in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
    profile[c] = profile[c].round(4)
profile.to_csv("outputs/hub_profile.csv", index=False)

# scale & KMeans K=3
cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
X = profile[cols].values
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X_s)
profile["cluster"] = labels

# separation evidence for K=2..5
def across_over_within(Xs, labs):
    n = len(Xs)
    across, within = [], []
    for i, j in combinations(range(n), 2):
        d = np.linalg.norm(Xs[i] - Xs[j])
        if labs[i] == labs[j]:
            within.append(d)
        else:
            across.append(d)
    return float(np.mean(across) / np.mean(within)) if within else np.nan

sep_rows = []
for K in [2, 3, 4, 5]:
    km_k = KMeans(n_clusters=K, random_state=42, n_init=10)
    labs = km_k.fit_predict(X_s)
    ratio = across_over_within(X_s, labs)
    sizes = sorted(np.bincount(labs), reverse=True)
    sep_rows.append({
        "K": K,
        "across_over_within": round(ratio, 4),
        "cluster_sizes": "-".join(str(s) for s in sizes),
    })
pd.DataFrame(sep_rows).to_csv("outputs/part_d_separation.csv", index=False)

# name segments by mean delay_rate
cluster_delay = profile.groupby("cluster")["delay_rate"].mean().sort_values()
name_map = {cluster_delay.index[0]: "Standard",
            cluster_delay.index[1]: "Watch",
            cluster_delay.index[2]: "Escalate"}
profile["segment"] = profile["cluster"].map(name_map)

# hub_segments
hub_info = hubs.set_index("hub_id")[["hub_city", "hub_tier"]]
seg = profile[["hub_id", "segment"]].copy()
seg = seg.join(hub_info, on="hub_id")
seg = seg.sort_values("hub_id").reset_index(drop=True)
seg.to_csv("outputs/hub_segments.csv", index=False)

# segment_profile
seg_prof = profile.groupby("segment").agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean"),
).reset_index()
# order Standard, Watch, Escalate
order = {"Standard": 0, "Watch": 1, "Escalate": 2}
seg_prof = seg_prof.sort_values("segment", key=lambda s: s.map(order)).reset_index(drop=True)
for c in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
    seg_prof[c] = seg_prof[c].round(4)
seg_prof.to_csv("outputs/segment_profile.csv", index=False)

# tier vs segment mismatch
tier_expect = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = seg[seg.apply(lambda r: r["segment"] != tier_expect[r["hub_tier"]], axis=1)].copy()
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print("Part D segments done.")
print(seg_prof.to_string(index=False))
print("mismatches:")
print(mismatch.to_string(index=False))
print("separation:")
print(pd.DataFrame(sep_rows).to_string(index=False))
