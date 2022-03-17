import numpy as np
from helpers import *
class ekf2:
    def __init__(self, numLandmarks, K,b,imu_T_cam,) -> None:
        
        self.numLandmarks = numLandmarks


        # IMU mean and covar
        self.IMUpose = np.eye(4) #4x4
        IMUsigInit = 100*np.eye(6) #6x6

        # Landmark mean and combo covar
        self.LMpose = np.zeros((numLandmarks, 3)) #landmarkposes Mx3 -> 3M
        PriorLandMarkSig = 50 * np.eye(3) 
        # 3M+6x3M+6 
        self.COMBOsig = np.block([[IMUsigInit, np.zeros((6,3*numLandmarks))],[np.zeros((3*numLandmarks,6)), np.kron(np.eye(numLandmarks), PriorLandMarkSig)]])

        # measurement noise covar
        self.V = 100 * np.eye(4) # observation noise
        # self.W = np.diag([0.3,0.3,0.3,0.05,0.05,0.05]) # motion noise
        self.W = 10000000000*np.eye(6) # motion noise


        #camera
        self.iTo = imu_T_cam
        self.oTi = np.linalg.inv(imu_T_cam)
        self.K = K
        self.b=b
        self.fsub = self.K[0,0]*self.b
        self.cv = self.K[1,2]
        self.fsv = self.K[1,1]
        self.cu = self.K[0,2]
        self.fsu = self.K[0,0]
        self.Ks = np.array([[self.fsu,0,self.cu,0],[0,self.fsv,self.cv,0],[self.fsu,0,self.cu,-self.fsub],[0,self.fsv,self.cv,0]])

        #other
        self.initializedAlready = np.zeros((numLandmarks), dtype=bool) # bool array to store if landmark is visited
        self.numOfLMInitialized = 0
        self.maxInitIdx = 0
        # self.landmarkmeans={i:[] for i in range(numLandmarks)}
        self.landmarkmeans={}

        self.Visitedlandmarkset = set()
        self.world=[]
        self.PT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64).T

    @property
    def oTw(self):
       return self.oTi @ np.linalg.inv(self.IMUpose)
    def predict(self,w,v,tau):
        self.IMUpose = self.IMUpose @ expm(tau*hat2(w,v))

        exp = expm(-tau*hat3(w,v)) #6x6
        # self.IMUsig[:6,:] = exp @ self.IMUsig[:6,:]
        # self.IMUsig[:,:6] = self.IMUsig[:,:6] @ exp.T
        # self.IMUsig[:6,:6] += self.W 
        # self.IMUsig[:6,:6] = exp @ self.IMUsig[:6,:6] @ exp.T + self.W 

        # self.LMsig
        # self.LMsig[:6,:6] = exp @ self.LMsig[:6,:6] @ exp.T + self.W 
        self.COMBOsig[:6,:] = exp @ self.COMBOsig[:6,:]
        self.COMBOsig[:,:6] = self.COMBOsig[:,:6] @ exp.T
        self.COMBOsig[:6,:6] += self.W 
    
    def NewLandmarks(self,f,validIdxs): #Initialize unseen landmarks
        tmp = np.invert(self.initializedAlready[validIdxs]) #fast set
        validIdxs = validIdxs[tmp]

        f = f[:, validIdxs]
        NewObs = validIdxs.size
        if NewObs > 0: # give them a coord
            self.initializedAlready[validIdxs]=True
            ul,vl,ur,vr = f[0],f[1],f[2],f[3]
            z = self.fsub / (ul-ur)
            y = z*(vl-self.cv)/self.fsv
            x = z*(ul-self.cu)/self.fsu
            sb = np.ones(f.shape) #4 by N
            sb[0],sb[1],sb[2] = x,y,z
            sw = np.linalg.inv(self.oTw)@sb
            self.LMpose[validIdxs] = (sw[:3]).T
            self.numOfLMInitialized+=NewObs
            self.maxInitIdx = max(self.maxInitIdx, np.max(validIdxs)+1) # +1 for index range python cutoff
            # return sw
        # else:
            # return 0
    def getH(self, f, validIdxs, Nt): #validIdxs also updates old ones
        LMposes = np.append(self.LMpose[validIdxs], np.ones((Nt,1)), axis=1)
        # more dynamic H (optimize)
        # H = np.zeros((Nt * 4, self.maxInitIdx * 3))
        # dynamic H
        H = np.zeros((Nt * 4, self.maxInitIdx*3 +6)) # +6 in SLAM

        for i in range(Nt):
            j = validIdxs[i]
            # 4x3
            H_update = self.Ks @ deriv_pi(self.oTw @ LMposes[i].reshape(4,1)) @ self.oTw @ self.PT
            # print("H",H_update.shape)
            H[i*4:(i+1)*4, j*3+6:(j+1)*3+6] = H_update #+6 to cols for slam
        return H
    def getK(self, H, Nt):
        n = self.maxInitIdx
        IxV = np.kron(np.eye(Nt), self.V)
        lmsig = self.COMBOsig[:n * 3 + 6, :n * 3 + 6] #+6
        sigHT = lmsig @ H.T
        HsigHT = H @ sigHT
        K = sigHT @ np.linalg.inv((HsigHT + IxV))
        return K, lmsig
    def getInnovation(self,f,validIdxs, Nt):
        z1 = f[:,validIdxs].reshape(-1,1,order='F')
        lmpose = np.hstack([self.LMpose[validIdxs, :], np.ones((Nt, 1))])
        z2 = (self.Ks @ canproj(self.oTw @ lmpose.T)).reshape(-1, 1, order='F')
        return z1-z2


    def update(self, f): #only update landmarks
        validIdxs = np.array(np.where(f.sum(axis=0) > -4), dtype=np.int32).reshape(-1)
        Nt = validIdxs.size #number of observations
        if Nt>0:
            self.NewLandmarks(f, validIdxs)
            H = self.getH(f, validIdxs, Nt)
            K, lmsig = self.getK(H,Nt)
            Innovation = self.getInnovation(f, validIdxs, Nt)

            #update LMu
            ut_1 = self.LMpose[:self.maxInitIdx] + (K@(Innovation))[6:,0].reshape(-1,3)
            self.LMpose[:self.maxInitIdx,:] = ut_1
            
            #update LMsig
            sig = (np.eye(K.shape[0]) - K@H) @ lmsig
            self.COMBOsig[:self.maxInitIdx*3 +6, :self.maxInitIdx*3 +6] = sig #+6

            #update IMUPose
            ut = self.IMUpose
            w = (K@(Innovation))[:3,0]
            v = (K@(Innovation))[3:6,0]
            self.IMUpose = ut @ expm(hat2(w,v))

    
    def getmeans(self,f): #use this instead of update to see landmarks without EKF updates
        # validIdxs = np.array(np.where(f.sum(axis=0) > -4), dtype=np.int32).reshape(-1)
        # #ONLY INSERT POSE IF VALID and havent seen (validIdxs2:valid and havent seen)
        # validIdxs2 = validIdxs[np.invert(self.initializedAlready[validIdxs])]
        # f = f[:,validIdxs2]
        # self.initializedAlready[validIdxs2]=True
        # ul,vl,ur,vr = f[0],f[1],f[2],f[3]
        # z = self.fsub / (ul-ur)
        # y = z*(vl-self.cv)/self.fsv
        # x = z*(ul-self.cu)/self.fsu
        # sb = np.ones(f.shape) #4 by N
        # sb[0],sb[1],sb[2] = x,y,z
        # sw = np.linalg.inv(self.oTw)@sb
        # self.LMpose[validIdxs2] = (sw[:3]).T

        validIdxs = np.array(np.where(f.sum(axis=0) > -4), dtype=np.int32).reshape(-1)
        #ONLY INSERT POSE IF VALID and havent seen (validIdxs2:valid and havent seen)
        validIdxs2 = [idx for idx in validIdxs if idx not in self.landmarkmeans]
        f = f[:,validIdxs2]
        self.initializedAlready[validIdxs2]=True
        ul,vl,ur,vr = f[0],f[1],f[2],f[3]
        z = self.fsub / (ul-ur)
        y = z*(vl-self.cv)/self.fsv
        x = z*(ul-self.cu)/self.fsu
        sb = np.ones(f.shape) #4 by N
        sb[0],sb[1],sb[2] = x,y,z
        sw = np.linalg.inv(self.oTw)@sb
        # self.LMpose[validIdxs2] = (sw[:3]).T
        for i in range(sw.shape[1]):
            self.landmarkmeans[validIdxs2[i]] = (sw[0,i],sw[1,i])

    
