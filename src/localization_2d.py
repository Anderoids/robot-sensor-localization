import numpy as np
import matplotlib.pyplot as plt

time = np.arange(0, 101, 1)

x_true = np.zeros(len(time))
y_true = np.zeros(len(time))

vx = 1.0
vy = 0.5

for i in range(1, len(time)):

    dt = time[i] - time[i - 1]

    x_true[i] = x_true[i - 1] + vx * dt
    y_true[i] = y_true[i - 1] + vy * dt

rng=np.random.default_rng(42)
gps_x=x_true+rng.normal(0,2,len(time))
gps_y=y_true+rng.normal(0,2,len(time))

plt.figure(figsize=(8, 6))

# Actual robot trajectory
plt.plot(
    x_true,
    y_true,
    label="True Robot Path"
)

# GPS measurements
plt.scatter(
    gps_x,
    gps_y,
    s=15,
    alpha=0.5,
    label="GPS Measurements"
)

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("2D Robot Localization with GPS Noise")
plt.legend()
plt.grid()

plt.show()

error_x = gps_x - x_true
error_y = gps_y - y_true

position_error = np.sqrt(
    error_x**2 + error_y**2
)

print("Mean 2D GPS error:", np.mean(position_error))
print("Max 2D GPS error:", np.max(position_error))

plt.figure(figsize=(10, 4))

plt.plot(time, position_error)

plt.xlabel("Time (seconds)")
plt.ylabel("Position Error (m)")
plt.title("2D GPS Position Error")
plt.grid()

plt.show()

#Turning robot simulation

x=np.zeros(len(time))
y=np.zeros(len(time))
theta=np.zeros(len(time))

v=1.0 #forward speed(m/s)
omega=0.05 #angular velocity (rad/s)

for i in range(1,len(time)):
    dt=time[i]-time[i-1]

    x[i]=x[i-1]+v*np.cos(theta[i-1])*dt
    y[i]=y[i-1]+v*np.sin(theta[i-1])*dt

    theta[i]=theta[i-1]+omega*dt


plt.figure(figsize=(7, 7))

plt.plot(x, y, label="True Robot Path")

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Turning Robot Trajectory")
plt.axis("equal")
plt.grid()
plt.legend()

plt.show()

x_dr = np.zeros(len(time))
y_dr = np.zeros(len(time))
theta_dr = np.zeros(len(time))

estimated_v = 1.03
estimated_omega = 0.048

for i in range(1, len(time)):
    dt = time[i] - time[i-1]

    x_dr[i] = x_dr[i-1] + estimated_v * np.cos(theta_dr[i-1]) * dt
    y_dr[i] = y_dr[i-1] + estimated_v * np.sin(theta_dr[i-1]) * dt

    theta_dr[i] = theta_dr[i-1] + estimated_omega * dt



plt.figure(figsize=(7, 7))

plt.plot(x, y, label="True Path")
plt.plot(x_dr, y_dr, "--", label="Dead Reckoning")

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("2D Dead Reckoning Drift")
plt.axis("equal")
plt.grid()
plt.legend()

plt.show()

rng = np.random.default_rng(42)

gps_x = x + rng.normal(0, 2, len(time))
gps_y = y + rng.normal(0, 2, len(time))

x_fused = np.zeros(len(time))
y_fused = np.zeros(len(time))
theta_fused = np.zeros(len(time))

gps_weight = 0.2

for i in range(1, len(time)):
    dt = time[i] - time[i-1]

    # Predict using dead reckoning
    theta_fused[i] = theta_fused[i-1] + estimated_omega * dt

    x_pred = x_fused[i-1] + estimated_v * np.cos(theta_fused[i-1]) * dt
    y_pred = y_fused[i-1] + estimated_v * np.sin(theta_fused[i-1]) * dt

    # Correct position using GPS
    x_fused[i] = (1-gps_weight) * x_pred + gps_weight * gps_x[i]
    y_fused[i] = (1-gps_weight) * y_pred + gps_weight * gps_y[i]


plt.figure(figsize=(8, 8))

plt.plot(x, y, label="True Path")
plt.plot(x_dr, y_dr, "--", label="Dead Reckoning")
plt.scatter(gps_x, gps_y, s=10, alpha=0.4, label="GPS")
plt.plot(x_fused, y_fused, label="Fused Estimate")

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("2D Robot Localization")
plt.axis("equal")
plt.grid()
plt.legend()

plt.show()

dr_error = np.sqrt(
    (x_dr - x)**2 +
    (y_dr - y)**2
)

gps_error = np.sqrt(
    (gps_x - x)**2 +
    (gps_y - y)**2
)

fused_error = np.sqrt(
    (x_fused - x)**2 +
    (y_fused - y)**2
)

dr_rmse = np.sqrt(np.mean(dr_error**2))
gps_rmse = np.sqrt(np.mean(gps_error**2))
fused_rmse = np.sqrt(np.mean(fused_error**2))

print("Dead Reckoning RMSE:", dr_rmse)
print("GPS RMSE:", gps_rmse)
print("Fused RMSE:", fused_rmse)