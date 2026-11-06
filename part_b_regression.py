"""Part B — Delivery-Hours Regression."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES].copy()
y = df["delivery_hours"].copy()

# two-step split → 840 / 180 / 180
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train_rows, val_rows, test_rows = len(X_train), len(X_val), len(X_test)

# scaling without leakage
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

train_scaled_max_abs_mean = float(np.abs(X_train_s.mean(axis=0)).max())
test_scaled_max_abs_mean = float(np.abs(X_test_s.mean(axis=0)).max())

pd.DataFrame([
    ("train_rows", train_rows),
    ("val_rows", val_rows),
    ("test_rows", test_rows),
    ("train_scaled_max_abs_mean", train_scaled_max_abs_mean),
    ("test_scaled_max_abs_mean", round(test_scaled_max_abs_mean, 4)),
], columns=["metric", "value"]).to_csv("outputs/part_b_checks.csv", index=False)

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return round(mae, 4), round(mse, 4), round(rmse, 4), round(r2, 4)

# baseline
mean_pred = np.full(len(y_val), y_train.mean())
b_mae, b_mse, b_rmse, b_r2 = metrics(y_val, mean_pred)

# tree depth diagnostic
depths = [1, 2, 3, 4, 6, 8, 12, None]
tree_rows = []
best_depth, best_val_rmse = None, 1e9
for d in depths:
    m = DecisionTreeRegressor(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    tr = np.sqrt(mean_squared_error(y_train, m.predict(X_train)))
    vr = np.sqrt(mean_squared_error(y_val, m.predict(X_val)))
    label = "unlimited" if d is None else d
    tree_rows.append({"max_depth": label, "train_rmse": round(tr, 4), "val_rmse": round(vr, 4)})
    if vr < best_val_rmse - 1e-9 or (abs(vr - best_val_rmse) < 1e-9 and (best_depth is None or (d is not None and (best_depth == "unlimited" or d < best_depth)))):
        best_val_rmse = vr
        best_depth = d
pd.DataFrame(tree_rows).to_csv("outputs/part_b_tree_depth.csv", index=False)

# four candidates
pipe_lr = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
pipe_lr.fit(X_train, y_train)
lr_mae, lr_mse, lr_rmse, lr_r2 = metrics(y_val, pipe_lr.predict(X_val))

dt = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
dt.fit(X_train, y_train)
dt_mae, dt_mse, dt_rmse, dt_r2 = metrics(y_val, dt.predict(X_val))

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_mae, rf_mse, rf_rmse, rf_r2 = metrics(y_val, rf.predict(X_val))

gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_mae, gb_mse, gb_rmse, gb_r2 = metrics(y_val, gb.predict(X_val))

# learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    m = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    m.fit(X_train, y_train)
    pred = m.predict(X_val)
    lr_rows.append({
        "learning_rate": lr,
        "val_rmse": round(np.sqrt(mean_squared_error(y_val, pred)), 4),
        "val_r2": round(r2_score(y_val, pred), 4),
    })
pd.DataFrame(lr_rows).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# rank by val RMSE
candidates = {
    "linear_regression": (lr_rmse, pipe_lr, lr_r2),
    "decision_tree": (dt_rmse, dt, dt_r2),
    "random_forest": (rf_rmse, rf, rf_r2),
    "gradient_boosting": (gb_rmse, gb, gb_r2),
}
ranked = sorted(candidates.items(), key=lambda x: x[1][0])
best_name = ranked[0][0]
second_name = ranked[1][0]

# 5-fold CV on train+val (1020 rows)
X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_rows = []
for name in [best_name, second_name]:
    if name == "linear_regression":
        model = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    elif name == "decision_tree":
        model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
    elif name == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    scores = cross_val_score(model, X_tv, y_tv, cv=kf, scoring="r2")
    cv_rows.append({"model": name, "mean_r2": round(scores.mean(), 4), "std_r2": round(scores.std(), 4)})
pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)

# final test score (model already fitted on train only)
final_model = candidates[best_name][1]
if best_name == "linear_regression":
    test_pred = final_model.predict(X_test)
else:
    test_pred = final_model.predict(X_test)
t_mae, t_mse, t_rmse, t_r2 = metrics(y_test, test_pred)

# metrics table
metric_rows = [
    {"model": "mean_baseline", "split": "val", "MAE": b_mae, "MSE": b_mse, "RMSE": b_rmse, "R2": b_r2},
    {"model": "linear_regression", "split": "val", "MAE": lr_mae, "MSE": lr_mse, "RMSE": lr_rmse, "R2": lr_r2},
    {"model": "decision_tree", "split": "val", "MAE": dt_mae, "MSE": dt_mse, "RMSE": dt_rmse, "R2": dt_r2},
    {"model": "random_forest", "split": "val", "MAE": rf_mae, "MSE": rf_mse, "RMSE": rf_rmse, "R2": rf_r2},
    {"model": "gradient_boosting", "split": "val", "MAE": gb_mae, "MSE": gb_mse, "RMSE": gb_rmse, "R2": gb_r2},
    {"model": best_name, "split": "test", "MAE": t_mae, "MSE": t_mse, "RMSE": t_rmse, "R2": t_r2},
]
pd.DataFrame(metric_rows).to_csv("outputs/part_b_metrics.csv", index=False)

print("Part B done. Best:", best_name, "val_rmse", ranked[0][1][0], "test_r2", t_r2)
print("best_depth", best_depth)
print("CV:", cv_rows)
