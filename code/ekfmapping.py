from pyexpat import features
import numpy as np
from helpers import *
class ekfmapping:
    def __init__(self, numLandmarks, K,b,imu_T_cam,
                 process_noise_covariance=None,
                 observation_noise_covariance=None,
                 prior_pose_covariance=None,
                 prior_landmark_covariance=None) -> None:

        if prior_landmark_covariance is None:
            prior_landmark_covariance = 5e-3 * np.eye(3)
        if prior_pose_covariance is None:
            prior_pose_covariance = 1e-3 * np.eye(6)
            # prior_pose_covariance = np.diag([0.3,0.3,0.3,0.05,0.05,0.05])
        if observation_noise_covariance is None:
            # observation_noise_covariance = 100 * np.eye(4)
            observation_noise_covariance = 1 * np.eye(4)
        if process_noise_covariance is None:
            process_noise_covariance = 1e-3 * np.eye(6)
        
        #stuff
        self.P = np.kron(np.eye(numLandmarks), prior_landmark_covariance)
        self.V = observation_noise_covariance
        self.W = process_noise_covariance
        self.xU=np.eye(4) #IMU pose
        self.xm = np.zeros((numLandmarks, 3)) #landmarkposes

        #other
        self.numLandmarks = numLandmarks
        self._initialized_mask = np.zeros((numLandmarks), dtype=bool)
        self._n_initialized = 0
        self._initialized_maxid = 0

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
        self.M=self.Ks
    
    @property
    def oTw(self):
        return self.oTi @ np.linalg.inv(self.xU)
    @property
    def n_initialized(self):
        return self._n_initialized

    def predict(self, w,v, tau):
        self.xU = self.xU @ expm(tau*hat2(w,v))

    def _make_zmap(self, z):
        assert z.ndim == 2 and z.shape[0] == 4
        # print(np.array(np.where(z.sum(axis=0) > -4)))
        return np.array(np.where(z.sum(axis=0) > -4), dtype=np.int32).reshape(-1)

    def _init_landmark(self, z, zmap):
        mask = np.invert(self._initialized_mask[zmap])
        zmap = zmap[mask]
        if zmap.size > 0:
            wTo = np.linalg.inv(self.oTw)
            self._initialized_mask[zmap] = True
            z = z[:, zmap]

            M = self.M
            b = self.b
            wcoord = np.ones((4, zmap.size))
            wcoord[0, :] = (z[0, :] - M[0, 2]) * b / (z[0, :] - z[2, :])
            wcoord[1, :] = (z[1, :] - M[1, 2]) * (-M[2, 3]) / (M[1, 1] * (z[0, :] - z[2, :]))
            wcoord[2, :] = -M[2, 3] / (z[0, :] - z[2, :])
            wcoord = wTo @ wcoord
            self.xm[zmap, :] = wcoord[:3, :].T

            self._n_initialized = np.sum(self._initialized_mask)
            self._initialized_maxid = max(zmap.max() + 1, self._initialized_maxid)

    def getH(self, z, zmap):
        n_observations = zmap.size
        n_updates = self._initialized_maxid

        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)

        xm = np.hstack([self.xm[zmap, :], np.ones((n_observations, 1))])
        H = np.zeros((n_observations * 4, n_updates * 3))

        for i in range(n_observations):
            obi = zmap[i]
            H[i * 4:(i + 1) * 4,
              obi * 3:(obi + 1) * 3] = self.Ks @ dpidq(self.oTw @ xm[i, :].reshape(-1, 1)) @ self.oTw @ P.T
        return H
    def _make_xm_P(self, z, zmap):
        n_observations = zmap.size
        n_updates = self._initialized_maxid

        xm = self.xm[:n_updates, :]
        P = self.P[:n_updates * 3, :n_updates * 3]

        return xm, P

    def _make_z(self, z, zmap):
        return z[:, zmap].reshape(-1, 1, order='F')

    def _make_predicted_z(self, z, zmap):
        n_observations = zmap.size

        xm = np.hstack([self.xm[zmap, :], np.ones((n_observations, 1))])
        zp = self.Ks @ pi(self.oTw @ xm.T)

        return zp.reshape(-1, 1, order='F')

    def _update_value_xm_P(self, xm, P, zmap):
        n_observations = zmap.size
        n_updates = self._initialized_maxid

        self.xm[:n_updates, :] = xm
        self.P[:n_updates * 3, :n_updates * 3] = P

    def update(self, features): # features at time t
        z = features
        zmap = self._make_zmap(z)
        print("zmapmx", np.max(zmap))
        # print(zmap)
        if zmap.size > 0:
            n_observations = zmap.size
            self._init_landmark(z, zmap)
            H = self.getH(z, zmap)
            xm, P = self._make_xm_P(z, zmap) #P is sigma
            zp = self._make_predicted_z(z, zmap)
            z = self._make_z(z, zmap)

            V = np.kron(np.eye(n_observations), self.V)
            PHT = P @ H.T
            K = np.linalg.solve((H @ PHT + V).T, PHT.T).T

            # print("WOOOO", z.shape, zp.shape)
            xm += (K @ (z - zp)).reshape(-1, 3)
            P = (np.eye(K.shape[0]) - K @ H) @ P

            self._update_value_xm_P(xm, P, zmap)


        
        
        
        
        # # zmap = np.array(np.where(z.sum(axis=0) > -4), dtype=np.int32).reshape(-1) # first has 13
        # zmap = self._make_zmap(z)
        # if zmap.size>0:
        #     Nt = zmap.size
        # # validIdxs = np.array(np.where(features.sum(axis=0) > -4), dtype=np.int32).reshape(-1) # first has 13
        # # Nt = validIdxs.size # number of valid obs
        # # if Nt>0:
        #     #init landmark z:feature, zmap:valididxs
        #     # mask = np.invert(self._initialized_mask[validIdxs])
        #     # validfeatures = validIdxs[mask]
        #     # z = features
        #     # zmap = validfeatures
        #     # if validfeatures.size>0:
        #     #     self._initialized_mask[validfeatures] = True
        #     #     z = features[:, validfeatures]
        #     #     zmap = validfeatures
        #     #     M = self.Ks
        #     #     b = self.b
        #     #     wto = np.linalg.inv(self.oTw)
        #     #     wcoord = np.ones((4, zmap.size))
        #     #     wcoord[0, :] = (z[0, :] - M[0, 2]) * b / (z[0, :] - z[2, :])
        #     #     wcoord[1, :] = (z[1, :] - M[1, 2]) * (-M[2, 3]) / (M[1, 1] * (z[0, :] - z[2, :]))
        #     #     wcoord[2, :] = -M[2, 3] / (z[0, :] - z[2, :])
        #     #     wcoord = wto @ wcoord
        #     #     self.xm[zmap, :] = wcoord[:3, :].T

        #     #     self._n_initialized = np.sum(self._initialized_mask)
        #     #     self._initialized_maxid = max(zmap.max() + 1, self._initialized_maxid)
            
        #     self._init_landmark(z, zmap)

        #     H = self.getH(z, zmap)
        #     xm, P = self._make_xm_P(z, zmap)
        #     zp = self._make_predicted_z(z, zmap)
        #     z = self._make_z(z, zmap)

        #     V = np.kron(np.eye(Nt), self.V)
        #     PHT = P @ H.T
        #     K = np.linalg.solve((H @ PHT + V).T, PHT.T).T

        #     xm += (K @ (z - zp)).reshape(-1, 3)
        #     P = (np.eye(K.shape[0]) - K @ H) @ P

        #     self._update_value_xm_P(xm, P, zmap)