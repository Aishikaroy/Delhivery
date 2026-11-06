"""Part C — Delay Classifier and Alert Threshold."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES].copy()
y = df["is_delayed"].copy()

# stratified two-step split → 840 / 180 / 180
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("delayed counts:", int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))

def conf_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return int(tn), int(fp), int(fn), int(tp), round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)

# --- models at default 0.5 ---
model_rows = []

# blind baseline: always not delayed
y_blind = np.zeros(len(y_val), dtype=int)
tn, fp, fn, tp, acc, prec, rec, f1 = conf_metrics(y_val, y_blind)
model_rows.append({"model": "blind_all_on_time", "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                   "accuracy": acc, "precision": 0, "recall": rec, "f1": f1})

# logistic
pipe_lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))])
pipe_lr.fit(X_train, y_train)
tn, fp, fn, tp, acc, prec, rec, f1 = conf_metrics(y_val, pipe_lr.predict(X_val))
model_rows.append({"model": "logistic_regression", "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                   "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

# decision tree — pick best depth by val F1
best_dt_f1, best_d = -1, 2
for d in [2, 3, 4, 5, 6, 8, 10]:
    m = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    f1 = f1_score(y_val, m.predict(X_val), zero_division=0)
    if f1 > best_dt_f1 + 1e-9 or (abs(f1 - best_dt_f1) < 1e-9 and d < best_d):
        best_dt_f1 = f1
        best_d = d
dt = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42)
dt.fit(X_train, y_train)
tn, fp, fn, tp, acc, prec, rec, f1 = conf_metrics(y_val, dt.predict(X_val))
model_rows.append({"model": f"decision_tree_depth_{best_d}", "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                   "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

# default RF
rf0 = RandomForestClassifier(n_estimators=100, random_state=42)
rf0.fit(X_train, y_train)
tn, fp, fn, tp, acc, prec, rec, f1 = conf_metrics(y_val, rf0.predict(X_val))
model_rows.append({"model": "random_forest_default", "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                   "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

pd.DataFrame(model_rows).to_csv("outputs/part_c_models.csv", index=False)

# --- feature scores ---
rf_imp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_imp.fit(X_train, y_train)
importances = rf_imp.feature_importances_

skb = SelectKBest(f_classif, k="all")
skb.fit(X_train, y_train)
f_scores = skb.scores_

feat_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": np.round(f_scores, 4),
    "rf_importance": np.round(importances, 4),
})
feat_df["f_rank"] = feat_df["f_score"].rank(ascending=False, method="min").astype(int)
feat_df["is_noise"] = feat_df["feature"].isin(NOISE_COLUMNS)
feat_df = feat_df.sort_values("f_rank").reset_index(drop=True)
feat_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

# selectk
selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([
        ("sel", SelectKBest(f_classif, k=k)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    f1 = f1_score(y_val, pipe.predict(X_val), zero_division=0)
    selectk_rows.append({"k": k, "val_f1": round(f1, 4)})
pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)

# from here k=7
selector = SelectKBest(f_classif, k=7)
X_train_7 = selector.fit_transform(X_train, y_train)
X_val_7 = selector.transform(X_val)
X_test_7 = selector.transform(X_test)

# tuning
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train_7, y_train)
grid_best = grid.best_estimator_
grid_val_f1 = f1_score(y_val, grid_best.predict(X_val_7), zero_division=0)

rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=9, cv=5, random_state=42, n_jobs=-1)
rand.fit(X_train_7, y_train)
rand_best = rand.best_estimator_
rand_val_f1 = f1_score(y_val, rand_best.predict(X_val_7), zero_division=0)

# default forest on same 7 features
def_rf = RandomForestClassifier(n_estimators=100, random_state=42)
def_rf.fit(X_train_7, y_train)
def_val_f1 = f1_score(y_val, def_rf.predict(X_val_7), zero_division=0)

tuning_rows = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid.best_params_),
        "best_cv_accuracy": round(grid.best_score_, 4),
        "val_f1": round(grid_val_f1, 4),
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(rand.best_params_),
        "best_cv_accuracy": round(rand.best_score_, 4),
        "val_f1": round(rand_val_f1, 4),
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": "",
        "val_f1": round(def_val_f1, 4),
    },
]
pd.DataFrame(tuning_rows).to_csv("outputs/part_c_tuning.csv", index=False)

# pick highest val_f1 (ties → default → grid → randomized)
candidates = [
    ("default_forest_same_7_features", def_val_f1, {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}),
    ("grid_search", grid_val_f1, grid.best_params_),
    ("randomized_search", rand_val_f1, rand.best_params_),
]
# sort by val_f1 desc, then by preferred order
order = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
candidates.sort(key=lambda x: (-x[1], order[x[0]]))
chosen_from, chosen_f1, chosen_params = candidates[0]

# final pipeline
final_params = {
    "n_estimators": chosen_params.get("n_estimators", 100),
    "max_depth": chosen_params.get("max_depth", None),
    "min_samples_leaf": chosen_params.get("min_samples_leaf", 1),
    "random_state": 42,
}
final_pipe = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(**final_params)),
])
final_pipe.fit(X_train, y_train)

# thresholds
probs_val = final_pipe.predict_proba(X_val)[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
thr_rows = []
chosen_threshold = 0.1
for t in thresholds:
    pred = (probs_val >= t).astype(int)
    tn, fp, fn, tp, acc, prec, rec, f1 = conf_metrics(y_val, pred)
    thr_rows.append({
        "threshold": t, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
    })
    if rec >= 0.80:
        chosen_threshold = t
pd.DataFrame(thr_rows).to_csv("outputs/part_c_thresholds.csv", index=False)

# highest t with recall >= 0.80
chosen_threshold = 0.1
for row in reversed(thr_rows):
    if row["recall"] >= 0.80:
        chosen_threshold = row["threshold"]
        break

# one test evaluation
probs_test = final_pipe.predict_proba(X_test)[:, 1]
pred_test = (probs_test >= chosen_threshold).astype(int)
tn, fp, fn, tp, acc, prec, rec, f1 = conf_metrics(y_test, pred_test)
pd.DataFrame([{
    "threshold": chosen_threshold, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
}]).to_csv("outputs/part_c_final_test.csv", index=False)

md = final_params["max_depth"]
pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": final_params["n_estimators"],
    "max_depth": "unlimited" if md is None else md,
    "min_samples_leaf": final_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from,
}]).to_csv("outputs/part_c_final_config.csv", index=False)

print("Part C done. chosen_from:", chosen_from, "threshold:", chosen_threshold)
print("test F1:", f1, "recall:", rec, "precision:", prec)
print("top7 features:", feat_df.head(7)["feature"].tolist())
