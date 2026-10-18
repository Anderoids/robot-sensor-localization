import numpy as np 
import matplotlib.pyplot as plt

time=np.arange(0,101,1)

true_velocity=1.8
estimated_velocity=2.0

true_position=true_velocity*time
rng=np.random.default_rng(42)
gps_position = np.zeros(len(time))

# Good GPS: 0-30 seconds
gps_position[:31] = (
    true_position[:31]
    + rng.normal(0, 2, 31)
)

# Bad GPS: 31-70 seconds
gps_position[31:71] = (
    true_position[31:71]
    + rng.normal(0, 10, 40)
)

# Good GPS again: 71-100 seconds
gps_position[71:] = (
    true_position[71:]
    + rng.normal(0, 2, 30)
)
kalman_position=np.zeros(len(time))
kalman_position[0]=0.0
P=1.0 #position variance

Q = 0.5
kalman_gain_history = np.zeros(len(time))

for i in range(1, len(time)):

    dt = time[i] - time[i - 1]

    # Predict position
    predicted_position = (
        kalman_position[i - 1]
        + estimated_velocity * dt
    )

    # Prediction uncertainty grows
    predicted_P = P + Q

    # GPS measurement uncertainty
    if 31 <= i <= 70:
        R = 100.0       # Bad GPS
    else:
        R = 4.0         # Good GPS

    # Kalman gain
    K = predicted_P / (predicted_P + R)
    kalman_gain_history[i] = K

    # Correct using GPS
    kalman_position[i] = (
        predicted_position
        + K * (gps_position[i] - predicted_position)
    )

    # Update uncertainty
    P = (1 - K) * predicted_P

plt.figure(figsize=(10, 5))

plt.plot(time, true_position, label="True Position")
plt.scatter(time, gps_position, s=10, alpha=0.4, label="GPS")
plt.plot(time, kalman_position, label="Kalman Estimate")

plt.xlabel("Time (seconds)")
plt.ylabel("Position (meters)")
plt.title("1D Kalman Filter Localization")
plt.legend()
plt.grid()
plt.show()

# Raw GPS error
gps_error = gps_position - true_position

# Kalman filter error
kalman_error = kalman_position - true_position

# Pure dead reckoning
dead_reckoning = estimated_velocity * time
dr_error = dead_reckoning - true_position

gps_rmse = np.sqrt(np.mean(gps_error**2))
kalman_rmse = np.sqrt(np.mean(kalman_error**2))
dr_rmse = np.sqrt(np.mean(dr_error**2))

print("GPS RMSE:", gps_rmse)
print("Dead Reckoning RMSE:", dr_rmse)
print("Kalman RMSE:", kalman_rmse)

plt.figure(figsize=(10, 4))

plt.plot(time, kalman_gain_history)

plt.axvspan(31, 70, alpha=0.15, label="Bad GPS")

plt.xlabel("Time (seconds)")
plt.ylabel("Kalman Gain K")
plt.title("Kalman Gain vs GPS Quality")
plt.legend()
plt.grid()

plt.show()