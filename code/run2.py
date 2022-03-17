from pr3_utils import *
from stereo import *
from helpers import *
from ekf import *
from visualize import *
filename = "./data/10.npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
# print(features.shape)
meanu = np.eye(4)
sig = 0.01*np.eye(6)
Wdef = np.diag([0.3,0.3,0.3,0.05,0.05,0.05])
pose = np.eye(4)
x=y=z=0
pred=[]
mxlandmarks = features.shape[1]
numLandmarks = mxlandmarks
stereo = stereo(K,b,imu_T_cam, numLandmarks)
ekf = ekf(stereo, numLandmarks)
# Jacobian
bigPose = np.zeros((4,4,len(t[0])))
bigPose[:,:,0] = pose
landmarks = []
l = []
for i in range(1,len(t[0])):
    print(i)
    linvel = linear_velocity[:,i]
    angvel = angular_velocity[:,i]
    
    # for n in numLandmarks:
    #     fs = features[:,n,i]ks


    # fs = features[:,:numLandmarks,i] # 4 by N
    fs = features[:,:,i] # 4 by N
    # fs = features[:,:,i][np.mod(np.arange(features[:,:,i].size),4)==0]
    

    # fs=fs.T
    # validIdxs = (fs[:,0] >0)
    # fs = fs[validIdxs]
    # print(fs.shape)
    # break
    # fs=fs.T
    
    # landmarkposeInW = stereo.getPoints3(fs, pose)

    landmarkposeInIMU = stereo.getPoints2(fs)
    # print(landmarkposeInIMU.shape)
    # print(landmarkposeInIMU[:,0])

    # break
    landmarkposeInW = pose @ landmarkposeInIMU
    # landmarkposeInW = pose @ np.linalg.inv(landmarkposeInIMU)
    # landmarkposeInW = pose @ landmarkposeInIMU 


    # plt.scatter(landmarkposeInW[0], landmarkposeInW[1])
    # plt.show(block=True)
    # break
    # ekf.updateMean(validIdxs,landmarkposeInW)
    landmarks.append(landmarkposeInW)
    # H = stereo.getH(pose, landmarkposeInW)
    # landmarks.append(landmarkposeInIMU)

    # if i==1266+1:
    if i==3000:
        plt.scatter(landmarkposeInW[0], landmarkposeInW[1])
        plt.show(block=True)
        break
    #     # print(fs.shape)
    #     print(fs[:,7])
    #     print(pose)
    #     print("\n")
    #     print(landmarkposeInIMU[:,7])
    #     print("\n")
    #     print(landmarkposeInW[:,7])
    #     break

    tau = t[0,i]-t[0,i-1]

    pose = getNextPose(angvel, linvel,  tau, pose)
    sig = getNextSig(angvel,linvel, tau, Wdef, sig)
    # x,y,z = pose[0,-1],pose[1,-1],pose[2,-1]
    bigPose[:,:,i] = pose

#     # pred.append((x,y))
# print(landmarkposeInW.shape)
# print(landmarkposeInW[:,0])
# for i, l in enumerate(landmarks):
    # for a in range(l.shape[1]):
        # if l[0,a]>10000:
            # print(l[0,a], i, a) #1266, 7
# print(len(landmarks))
# print(landmarks[1266][:,7])
# visualize_trajectory_2d(bigPose)
visualize_trajectory(bigPose, landmarks)
# visualize_trajectory(bigPose, l)




    



