from pr3_utils import *
from stereo import *
from helpers import *
from ekf import *
from visualize import *
filename = "./data/03.npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
# print(features.shape)
meanu = np.eye(4)
sig = 0.01*np.eye(6)
stereo = stereo(K,b,imu_T_cam)
Wdef = np.diag([0.3,0.3,0.3,0.05,0.05,0.05])
pose = np.eye(4)
x=y=z=0
prev = t[0,0]
pred=[]
numLandmarks=2
ekf = ekf(numLandmarks)
# Jacobian
bigPose = np.zeros((4,4,len(t[0])))
bigPose[:,:,0] = pose
landmarks = []
l = []
for i in range(1,len(t[0])):
    linvel = linear_velocity[:,i]
    angvel = angular_velocity[:,i]
    
    fs = features[:,:numLandmarks,i] # 4 by N
    # fs=fs.T
    # fs = fs[fs[:,0] >=0]
    # fs=fs.T
    # if fs.shape[1]==0:
    #     continue

    # fs = features[:,:,i]

    # plt.scatter(fs[0], fs[1])
    # plt.show(block = True)
    # break

    # tmp= stereo.getPoints(fs)
    # print(tmp)
    # l.append(tmp)
    landmarkposeInIMU = stereo.getPoints2(fs)
    # print(landmarkposeInIMU.shape)

    # plt.scatter(landmarkposeInIMU[0], landmarkposeInIMU[1])
    # plt.show(block = True)
    # break
    landmarkposeInW = pose @ landmarkposeInIMU
    landmarks.append(landmarkposeInW)
    break
    # if i==100: break
    # break
    # break


    # break

    tau = t[0,i]-prev
    prev = t[0,i]

    # S = np.concatenate((linear_velocity[:,i],angular_velocity[:,i]), axis=0)
    pose = getNextPose(angvel, linvel,  tau, pose)
    sig = getNextSig(angvel,linvel, tau, Wdef, sig)
    # print(NextPoseS)
    x,y,z = pose[0,-1],pose[1,-1],pose[2,-1]
    bigPose[:,:,i] = pose

#     # pred.append((x,y))

# visualize_trajectory_2d(bigPose)
visualize_trajectory(bigPose, landmarks)
# visualize_trajectory(bigPose, l)




    



