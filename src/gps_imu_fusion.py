import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

time = np.arange(0, 101, 1)
dt = 1.0

# Ground truth
true_acceleration = 0.02

true_velocity = np.zeros(len(time))
true_position = np.zeros(len(time))

for i in range(1, len(time)):

    true_velocity[i] = (
        true_velocity[i - 1]
        + true_acceleration * dt
    )

    true_position[i] = (
        true_position[i - 1]
        + true_velocity[i] * dt
    )


imu_bias = 0.01

imu_acceleration = (
    true_acceleration
    + imu_bias
    + rng.normal(0, 0.02, len(time))
)

gps_position = (
    true_position
    + rng.normal(0, 2, len(time))
)

gps_available = np.ones(len(time), dtype=bool)

gps_available[40:61] = False

gps_position[~gps_available] = np.nan

plt.figure(figsize=(10, 5))

plt.plot(time, true_position, label="True Position")
plt.scatter(time, gps_position, s=12, label="GPS")

plt.axvspan(40, 60, alpha=0.15, label="GPS Outage")

plt.xlabel("Time (seconds)")
plt.ylabel("Position (meters)")
plt.title("GPS Outage Simulation")
plt.legend()
plt.grid()

plt.show()

imu_velocity = np.zeros(len(time))
imu_position = np.zeros(len(time))

for i in range(1,len(time)):
    #accerlation->velocity
    imu_velocity[i]=(
        imu_velocity[i-1]+
        imu_acceleration[i]*dt
    )

    #velocity -> position

    imu_position[i]=(
        imu_position[i-1]+
        imu_velocity[i]*dt
    )


plt.figure(figsize=(10, 5))

plt.plot(time, true_position, label="True Position")
plt.plot(time, imu_position, label="IMU Position")

plt.scatter(
    time,
    gps_position,
    s=12,
    label="GPS"
)

plt.axvspan(
    40,
    60,
    alpha=0.15,
    label="GPS Outage"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Position (meters)")
plt.title("GPS Outage with IMU Dead Reckoning")
plt.legend()
plt.grid()

plt.show()


fused_position = np.zeros(len(time))
fused_velocity = np.zeros(len(time))

P = 1.0
Q = 0.5
R = 4.0

for i in range(1, len(time)):

    # -----------------
    # PREDICTION: IMU
    # -----------------

    fused_velocity[i] = (
        fused_velocity[i - 1]
        + imu_acceleration[i] * dt
    )

    predicted_position = (
        fused_position[i - 1]
        + fused_velocity[i] * dt
    )

    predicted_P = P + Q

        # -----------------
    # CORRECTION: GPS
    # -----------------

    if gps_available[i]:

        K = predicted_P / (predicted_P + R)

        fused_position[i] = (
            predicted_position
            + K * (gps_position[i] - predicted_position)
        )

        P = (1 - K) * predicted_P

    else:

        fused_position[i] = predicted_position
        P = predicted_P

plt.figure(figsize=(10, 5))

plt.plot(time, true_position, label="True Position")
plt.plot(time, fused_position, label="GPS + IMU Fusion")

plt.scatter(
    time,
    gps_position,
    s=12,
    alpha=0.5,
    label="GPS"
)

plt.axvspan(
    40,
    60,
    alpha=0.15,
    label="GPS Outage"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Position (meters)")
plt.title("GPS + IMU Fusion During GPS Outage")
plt.legend()
plt.grid()

plt.show()

imu_error=imu_position-true_position
fusion_error=fused_position-true_position

imu_rmse=np.sqrt(np.mean(imu_error**2))
fusion_rmse=np.sqrt(np.mean(fusion_error**2))

# GPS contains NaN during tunnel, so only evaluate where GPS exists
gps_error=gps_position[gps_available]-true_position[gps_available]
gps_rmse=np.sqrt(np.mean(gps_error**2))

print("GPS RMSE:", gps_rmse)
print("IMU RMSE:", imu_rmse)
print("GPS + IMU Fusion RMSE:", fusion_rmse)