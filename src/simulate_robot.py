import numpy as np
import matplotlib.pyplot as plt

time=np.arange(0,101,1) # moves from 0m to 100m

true_position=time # assuming the robot moves 1m every second
print(true_position)

#visualize
plt.plot(time,true_position)
plt.xlabel('Time (seconds)')
plt.ylabel('Postion (Meters)')
plt.title('Robot Ground Truth')
plt.show()

#adding GPS measurements
gps_position=true_position-3+ np.random.normal(0,2,len(time))
print(gps_position)

plt.plot(time,true_position,label='True_postion')
plt.plot(time,gps_position,label='GPS Measurement')
plt.xlabel('Time(seconds)')
plt.ylabel('Position (meters)')
plt.title('True position vs gps')
plt.legend()
plt.show()

#error 
error=gps_position-true_position
print(error)

plt.hist(error,bins=20)
plt.xlabel("GPS Error(meters)")
plt.ylabel('Frequency')
plt.title('Distribution of GPS Error')
plt.show()

mean_error=np.mean(error)
std_error=np.std(error)

print('Mean error: ',mean_error)
print('Standard error: ',std_error)

rmse=np.sqrt(np.mean(error**2))
print("RMSE: ",rmse)