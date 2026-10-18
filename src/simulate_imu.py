import numpy as np
import matplotlib.pyplot as plt

time = np.arange(0, 11, 1)

true_acceleration = 2.0

true_velocity = np.zeros(len(time))
true_position = np.zeros(len(time))

true_position=true_velocity*time

for i in range(1, len(time)):

    dt = time[i] - time[i - 1]

    # Acceleration -> Velocity
    true_velocity[i] = (
        true_velocity[i - 1]
        + true_acceleration * dt
    )

    # Velocity -> Position
    true_position[i] = (
        true_position[i - 1]
        + true_velocity[i] * dt
    )
print("Velocity:", true_velocity)
print("Position:", true_position)

rng=np.random.default_rng(42)

imu_bias = 0.2

imu_accerlation = (
    true_acceleration
    + imu_bias
    + rng.normal(0, 0.2, len(time))
)
print("IMU acceleration: ",imu_accerlation)

imu_velocity = np.zeros(len(time))
imu_position = np.zeros(len(time))

for i in range(1, len(time)):

    dt = time[i] - time[i - 1]

    # IMU acceleration -> estimated velocity
    imu_velocity[i] = (
        imu_velocity[i - 1]
        + imu_accerlation[i] * dt
    )

    # Estimated velocity -> estimated position
    imu_position[i] = (
        imu_position[i - 1]
        + imu_velocity[i] * dt
    )

plt.figure(figsize=(10, 5))

plt.plot(time, true_position, label="True Position")
plt.plot(time, imu_position, label="IMU Estimated Position")

plt.xlabel("Time (seconds)")
plt.ylabel("Position (meters)")
plt.title("Position Estimated from IMU")
plt.legend()
plt.grid()

plt.show()

imu_position_error = imu_position - true_position

print("Final true position:", true_position[-1])
print("Final IMU position:", imu_position[-1])
print("Final position error:", imu_position_error[-1])

imu_rmse = np.sqrt(np.mean(imu_position_error**2))

print("IMU Position RMSE:", imu_rmse)