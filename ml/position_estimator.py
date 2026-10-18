import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

time = np.arange(0, 501, 1)
dt = 1.0

x_true = np.zeros(len(time))
y_true = np.zeros(len(time))
theta_true = np.zeros(len(time))

v = 1.0
omega = 0.02

for i in range(1, len(time)):

    theta_true[i] = theta_true[i-1] + omega * dt

    x_true[i] = (
        x_true[i-1]
        + v * np.cos(theta_true[i-1]) * dt
    )

    y_true[i] = (
        y_true[i-1]
        + v * np.sin(theta_true[i-1]) * dt
    )

gps_x = x_true + rng.normal(0, 2, len(time))
gps_y = y_true + rng.normal(0, 2, len(time))


x_dr = np.zeros(len(time))
y_dr = np.zeros(len(time))
theta_dr = np.zeros(len(time))

estimated_v = 1.03
estimated_omega = 0.019

for i in range(1, len(time)):

    theta_dr[i] = theta_dr[i-1] + estimated_omega * dt

    x_dr[i] = (
        x_dr[i-1]
        + estimated_v * np.cos(theta_dr[i-1]) * dt
    )

    y_dr[i] = (
        y_dr[i-1]
        + estimated_v * np.sin(theta_dr[i-1]) * dt
    )

data = pd.DataFrame({
    "gps_x": gps_x,
    "gps_y": gps_y,
    "dr_x": x_dr,
    "dr_y": y_dr,
    "true_x": x_true,
    "true_y": y_true
})

print(data.head())



from sklearn.model_selection import train_test_split

X = data[["gps_x", "gps_y", "dr_x", "dr_y"]]
y = data[["true_x", "true_y"]]

split_index = int(0.8 * len(data))

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

pred_x = predictions[:, 0]
pred_y = predictions[:, 1]

true_x_test = y_test["true_x"].to_numpy()
true_y_test = y_test["true_y"].to_numpy()

ml_error = np.sqrt(
    (pred_x - true_x_test)**2 +
    (pred_y - true_y_test)**2
)

ml_rmse = np.sqrt(np.mean(ml_error**2))

print("ML Localization RMSE:", ml_rmse)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))

plt.plot(
    true_x_test,
    true_y_test,
    label="True Path",
    linewidth=3
)

plt.plot(
    pred_x,
    pred_y,
    "--",
    label="ML Estimate"
)

plt.scatter(
    X_test["gps_x"],
    X_test["gps_y"],
    s=15,
    alpha=0.4,
    label="GPS"
)

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("ML Robot Localization")
plt.axis("equal")
plt.grid()
plt.legend()

plt.show()

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_x = rf_predictions[:, 0]
rf_y = rf_predictions[:, 1]

rf_error = np.sqrt(
    (rf_x - true_x_test)**2 +
    (rf_y - true_y_test)**2
)

rf_rmse = np.sqrt(np.mean(rf_error**2))

print("Linear Regression RMSE:", ml_rmse)
print("Random Forest RMSE:", rf_rmse)

#gps baseline
gps_x_test = gps_x[split_index:]
gps_y_test = gps_y[split_index:]

gps_error_test = np.sqrt(
    (gps_x_test - true_x_test)**2 +
    (gps_y_test - true_y_test)**2
)

gps_rmse_test = np.sqrt(np.mean(gps_error_test**2))

#dead-reckoning baseline
dr_x_test = x_dr[split_index:]
dr_y_test = y_dr[split_index:]

dr_error_test = np.sqrt(
    (dr_x_test - true_x_test)**2 +
    (dr_y_test - true_y_test)**2
)

dr_rmse_test = np.sqrt(np.mean(dr_error_test**2))

#fusion
gps_weight = 0.2

fused_x_test = (
    (1 - gps_weight) * dr_x_test
    + gps_weight * gps_x_test
)

fused_y_test = (
    (1 - gps_weight) * dr_y_test
    + gps_weight * gps_y_test
)

fusion_error_test = np.sqrt(
    (fused_x_test - true_x_test)**2 +
    (fused_y_test - true_y_test)**2
)

fusion_rmse_test = np.sqrt(
    np.mean(fusion_error_test**2)
)

print("\n--- Localization Comparison ---")
print("GPS RMSE:", gps_rmse_test)
print("Dead Reckoning RMSE:", dr_rmse_test)
print("Static Fusion RMSE:", fusion_rmse_test)
print("Linear Regression RMSE:", ml_rmse)
print("Random Forest RMSE:", rf_rmse)

methods = [
    "GPS",
    "Linear Regression",
    "Random Forest",
    "Static Fusion",
    "Dead Reckoning"
]

rmse_values = [
    gps_rmse_test,
    ml_rmse,
    rf_rmse,
    fusion_rmse_test,
    dr_rmse_test
]

plt.figure(figsize=(9, 5))
plt.bar(methods, rmse_values)

plt.ylabel("Position RMSE (m)")
plt.title("Localization Method Comparison")
plt.xticks(rotation=20)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/localization_comparison.png", dpi=300)
plt.show()