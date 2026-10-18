import numpy as np
import matplotlib.pyplot as plt
time=np.arange(0,101,1)
velocity=1.0
true_velocity=1.8
estimated_velocity=2.0
true_postion=true_velocity*time

estimated_position=np.zeros(len(time))

for i in range(1,len(time)):
    dt=time[i]-time[i-1]

    estimated_position[i]=(
        estimated_position[i-1]+estimated_velocity*dt
    )

plt.plot(time,true_postion,label='true_position')
plt.plot(time,estimated_position,"--",label='Estimated position')

plt.xlabel("Time (seconds)")
plt.ylabel('Position (meters)')
plt.legend()
plt.show()

#simulate noisy GPS
rng=np.random.default_rng(42)

gps_position=np.zeros(len(time))
#Good GPS:sd=2m
gps_position[:31]=(
     true_postion[:31]+rng.normal(0,2,31)
)

#Bad GPS: sd=10m
gps_position[31:71]=(
     true_postion[31:71]+rng.normal(0,10,40)
)

#Good gps again
gps_position[71:]=(
     true_postion[71:]+rng.normal(0,2,30)
)
#Combine GPS and dead reckoning

fused_position=(
    0.95*gps_position+0.05*estimated_position
                )

plt.figure(figsize=(10,5))

plt.plot(time,true_postion,label="True position")
plt.plot(time,estimated_position,label="Dead reckoning")

plt.scatter(time,gps_position,label="GPS",s=8,alpha=0.5)

plt.plot(time,fused_position,label='Fused estimate')
plt.xlabel('Time (s)')
plt.ylabel("position (m)")
plt.title("GPS and Dead Reckoning")
plt.legend()
plt.grid()
plt.show()

gps_rmse=np.sqrt(
    np.mean((gps_position-true_postion)**2)
)
dr_rmse=np.sqrt(np.mean((estimated_position-true_postion)**2))

fused_rmse=np.sqrt(
    np.mean((fused_position-true_postion)**2)
)

print("GPS RMSE: ",gps_rmse)
print("Dead Reckoning RMSE: ",dr_rmse)
print("Fusion RMSE: ",fused_rmse)

#Simulate a GPS outrage from from 40 to 60 seconds
gps_available=np.ones(len(time),dtype=bool)

gps_available[40:61]=False

gps_with_outrage=gps_position.copy()
gps_with_outrage[~gps_available]=np.nan

plt.figure(figsize=(10, 5))

plt.plot(time, true_postion, label="True position")
plt.plot(time, estimated_position,
         label="Dead reckoning")

plt.scatter(time, gps_with_outrage,
            label="GPS", s=12)

plt.xlabel("Time (seconds)")
plt.ylabel("Position (meters)")
plt.legend()
plt.grid()
plt.show()


corrected_position=np.zeros(len(time))


for i in range(1,len(time)):
    dt=time[i]-time[i-1]

    predicted_position=( # predict using dr
        corrected_position[i-1]+
        estimated_velocity*dt
    )
    if 31<=i<=70:
         gps_weight=0.2
    else:
         gps_weight=0.8
    #convert using gps if avaliable

    if gps_available[i]:
        corrected_position[i]=(
            gps_weight*gps_position[i]+
            (1-gps_weight)*predicted_position
        )
    else:
        #if gps not available
            corrected_position[i]=predicted_position

plt.figure(figsize=(10, 5))

plt.plot(time, true_postion,
         label="True position")

plt.plot(time, estimated_position,
         label="Dead reckoning")

plt.scatter(time, gps_with_outrage,
            label="GPS", s=10)

plt.plot(time, corrected_position,
         label="Corrected estimate",
         linewidth=2)

plt.axvspan(40, 60, alpha=0.15,
            color="orange",
            label="GPS outage")

plt.xlabel("Time (seconds)")
plt.ylabel("Position (meters)")
plt.title("Robot Navigation During GPS Outage")
plt.legend()
plt.grid()
plt.show()

corrected_rmse=np.sqrt(
    np.mean((corrected_position-true_postion)**2)
)
print("Dead reckoning RMSE:", dr_rmse)
print("Corrected estimate RMSE:", corrected_rmse)

plt.figure(figsize=(10, 5))

plt.plot(time, true_postion, label="True position")
plt.plot(time, corrected_position, label="Adaptive estimate")
plt.scatter(time, gps_position, s=10, alpha=0.4, label="GPS")

plt.axvspan(31, 70, alpha=0.15,
            label="Bad GPS period")

plt.xlabel("Time (seconds)")
plt.ylabel("Position (meters)")
plt.title("Adaptive GPS + Dead Reckoning")
plt.legend()
plt.grid()
plt.show()

good_error = gps_position[:31] - true_postion[:31]
bad_error = gps_position[31:71] - true_postion[31:71]

good_std = np.std(good_error)
bad_std = np.std(bad_error)

print("Good GPS standard deviation:", good_std)
print("Bad GPS standard deviation:", bad_std)

prediction_std = 5.0

good_weight = (
    1 / good_std**2
) / (
    1 / good_std**2
    + 1 / prediction_std**2
)

bad_weight = (
    1 / bad_std**2
) / (
    1 / bad_std**2
    + 1 / prediction_std**2
)

print("Good GPS weight:", good_weight)
print("Bad GPS weight:", bad_weight)