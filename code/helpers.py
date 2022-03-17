import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

def hat1(x): # R3
    x1,x2,x3 = x
    return np.array([[0,-x3,x2],[x3,0,-x1],[-x2,x1,0]])
    # return np.array([[0,x3,x2],[-x3,0,-x1],[-x2,x1,0]])


def hat2(w,v): # zeta is w,v. R6 Twist, returns 4x4
    what = hat1(w)
    v=v.reshape(3,1)
    # print(p.shape, thetahat.shape)

    tmp1 = np.append(what, v, axis=1)
    tmp1 = np.append(tmp1, np.zeros((1,4)), axis=0)
    return tmp1
    # return np.block([[thetahat, p],[0,0]])

def hat3(w,v): #6x6 w and v are R3
    what=hat1(w)   
    tmp = np.append(what, hat1(v), axis=0)
    zero = np.zeros((3,3))
    tmp2 = np.append(zero, what, axis=0)
    return np.append(tmp,tmp2, axis=1)

def getNextR(Rk, tauk, w):
    what = hat1(w)
    return Rk @ expm(tauk*what)

def getNextT(Tk, tauk, w,v):
    zhat = hat2(w,v)
def canproj(q):
    return q / q[2, :]
def deriv_pi(q):
    q = q.reshape(-1)
    return (1/q[2]) * np.array([[1, 0, -q[0] / q[2], 0], [0, 1, -q[1] / q[2], 0], [0, 0, 0, 0], [0, 0, -q[3] / q[2], 1]])
def getNextPose(w,v, tau, prevPose):
    return prevPose@expm(tau*hat2(w,v))

def getNextSig(w,v, tau, W, sig):
    uhet = hat3(w,v)
    E = expm(-tau*uhet)
    return E@sig@E.T + W

def plotpred(pred):
    plt.scatter([x[0] for x in pred],[x[1] for x in pred])
    plt.show(block=True)

# zeta = np.array([1,2,3,4,5,6])

# # z = hat2(zeta)
# a = getPose(zeta)
# print(a)

