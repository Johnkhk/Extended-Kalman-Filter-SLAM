from pr3_utils import *
from visualize import *
from ekfmapping import *
from ekf_map import *
from ekf_slam import*
filename = "./data/10.npz"
# print("hello1")
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
print("hello2")
# print(features.shape, t.shape, linear_velocity.shape)

# Sample every 4th feature

everyth=20 #333 features
features = features[:,0:-1:everyth,:]
ekf = ekf2(features.shape[1], K,b,imu_T_cam)
# print("hello3")
# ekf = ekf(features.shape[1], K,b,imu_T_cam)
# print("hello4")

bigPose = np.zeros((4,4,len(t[0])))
bigPose[:,:,0] = np.eye(4)
pose = []
for i in range(1,len(t[0])):
    print(i)
    linvel = linear_velocity[:,i]
    angvel = angular_velocity[:,i]
    tau = t[0,i]-t[0,i-1]
    ekf.predict(angvel, linvel, tau) #updates xU
    # pose.append(np.linalg.inv(ekf.xU))
    pose.append(ekf.IMUpose)
    bigPose[:,:,i] = ekf.IMUpose
    ekf.update(features[:,:,i])
    # ekf.getmeans(features[:,:,i])
visualize_trajectory(bigPose, ekf.LMpose)
