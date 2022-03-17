import numpy as np
from helpers import *
class stereo:
    def __init__(self,K,b,imu_T_cam, numLandmarks) -> None:
        self.K = K # intrinsic calibration 3x3
        self.b = b # baseline

        self.T = imu_T_cam # 4X4 T from left camera to imu frame 
        self.Tinv = np.linalg.inv(imu_T_cam) # cam_T_imu
        # self.fsub = self.T[2,-1]
        self.fsub = self.K[0,0]*self.b
        # self.Ks = np.append(K, np.array([0,0,self.fsub]).reshape(3,1),axis=1)

        self.cv = self.K[1,2]
        self.fsv = self.K[1,1]
        self.cu = self.K[0,2]
        self.fsu = self.K[0,0]
        self.Ks = np.array([[self.fsu,0,self.cu,0],[0,self.fsv,self.cv,0],[self.fsu,0,self.cu,-self.fsub],[0,self.fsv,self.cv,0]])

        self.H = np.zeros((numLandmarks,4,3)) # Nx4x3

    def getPoints(self, f):
        ul,vl,ur,vr = f
        # print(self.K)
        z = self.fsub / (ul-ur)
        y = z*(vl-self.cv)/self.fsv
        x = z*(ul-self.cu)/self.fsu

        sw = np.array([x,y,z,np.ones(len(x))])
        sb = self.T@sw
        print(sb)

        x,y,z, _ = sb
        print(x,y,z)
        # return x,y,z
        return sb

    def getPoints3(self, f, pose): #4,5
        ul,vl,ur,vr = f[0],f[1],f[2],f[3]
        
        # print(self.K)
        # print(ul, ur)
        z = self.fsub / (ul-ur)
        y = z*(vl-self.cv)/self.fsv
        x = z*(ul-self.cu)/self.fsu
        # print(y.shape)
        # sw = np.array([x,y,z,1])]
        sw = np.zeros(f.shape) #4 by N
        sw[0],sw[1],sw[2],sw[3] = x,y,z,np.ones(f.shape[1])

        sb = np.linalg.inv(self.T)@np.linalg.inv(pose)@sw
        # print(sb)

        # x,y,z, _ = sb[0], sb[1], sb[2], sb[3]
        # print(x,y,z)
        # return x,y,z
        return sb # 4 by N

    def getPoints2(self, f): #4,5
        ul,vl,ur,vr = f[0],f[1],f[2],f[3]
        
        # print(self.K)
        # print(ul, ur)
        z = self.fsub / (ul-ur)
        y = z*(vl-self.cv)/self.fsv
        x = z*(ul-self.cu)/self.fsu
        # print(y.shape)
        # sw = np.array([x,y,z,1])]
        sw = np.zeros(f.shape) #4 by N
        sw[0],sw[1],sw[2],sw[3] = x,y,z,np.ones(f.shape[1])

        sb = self.T@sw
        # print(sb)

        # x,y,z, _ = sb[0], sb[1], sb[2], sb[3]
        # print(x,y,z)
        # return x,y,z
        return sb # 4 by N
    # def getJacobianH(self):
        # return H
    # def canonicalproj(self, x): #R3
    #     e = np.array([0,0,1]).reshape(3,x.shape[1])
    #     return (1/(e@x))@x
    def pi(self,q):
        assert q.ndim == 2 and q.shape[0] == 4
        return q / q[2, :]
    def getPredZ(self, m, pose):
        print("HI"*20)
        tmp = self.Tinv @ np.linalg.inv(pose) @ m
        
        # print(tmp.shape)
        return self.Ks @ self.pi(tmp)

    def getH(self, pose, u): # u is homogenous of landmark
        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        T = np.linalg.inv(pose)
        oTi = self.Tinv
        right = oTi@T@P.T
        # print(right.shape)
        tmp = oTi@T@u
        # print(tmp.shape)
        H=np.zeros((tmp.shape[1],4,3))
        for n in range(tmp.shape[1]):
            tmp2 = dpidq(tmp[:,n])
            # print(self.Ks.shape, tmp2.shape, right.shape)
            H[n] = self.Ks @ tmp2 @ right
        # return self.Ks @ tmp @ right
        return H

    

    # def getInnovation