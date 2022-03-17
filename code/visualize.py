import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from scipy.linalg import expm

def visualize_trajectory(pose,landmarks, path_name="path",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    # print("hi", type(landmarks))
    # print(len(landmarks))
    # xs = [x[0] for x in landmarks]
    # ys = [y[1] for y in landmarks]
    # xs = np.concatenate(xs).ravel().tolist()
    # ys = np.concatenate(ys).ravel().tolist()



    # landmarks = np.array(landmarks)
    # print(landmarks.shape)
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")

    #for landmarks
    # ax.scatter([x[0][0] for x in landmarks],[y[1][0] for y in landmarks])
    # ax.scatter(landmarks[:,0,:].flatten(), landmarks[:,1,:].flatten(), s=0.1)
    # ax.scatter([x[0] for x in landmarks[0]], [y[1] for y in landmarks[0]])
    # ax.scatter(xs,ys)
    # print(landmarks.shape)

    #working
    # ax.scatter([x for x in landmarks[:,0]], [y for y in landmarks[:,1]])
    
    #getmeans
    # xi,yi=[],[]
    # for k,v in landmarks.items():
    #     # landmarks[k] = mean(v)
    #     # xi.append([a[0][0] for a in v])
    #     # yi.append([a[0][1] for a in v])
    #     xi.append(v[0][0])
    #     yi.append(v[0][1])
    # ax.scatter(xi,yi)

    # ax.scatter([a[0] for a in landmarks.values()], [a[1] for a in landmarks.values()])

    # ax.scatter([a[0] for a in landmarks], [a[1] for a in landmarks])

    #Working with just (update, only init landmarks, use LMpose) (getmeans first block, use LMpose)
    # ax.scatter(landmarks[:,0], landmarks[:,1])
    ax.scatter(landmarks[:,0], landmarks[:,1], s=3, label="landmarks")
    ax.set_title("10.npz EKF Visual-Inertial SLAM (Skip 10 features)")
    #Visual-Inertial

  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)

    return fig, ax