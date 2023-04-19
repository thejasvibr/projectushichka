# -*- coding: utf-8 -*-
"""
Correspondence matching and 3d triangulation
============================================
Perform correspondence matching of tracks and then get 3D trajectories. 

Previously run module: traj_correction.py


Overall Observations (2023-03-20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Something's not quite right. The track2trajectory project needs some more 
documentation and testing in general. 

Conda Environment to activate: tproject


Notes
~~~~
* This module is built from a mix of quickexample.py and thermalcamera.py in the
tproject package examples folder. 
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from track2trajectory.camera import Camera
from track2trajectory.projection import project_to_2d_and_3d, calcFundamentalMatrix
from track2trajectory.match3d import match_2dpoints_to_3dtrajectories

#%% Load the camera 2D points
all_cam_2d = pd.read_csv('all_camera_tracks_2018-08-17_P01_7000_first25frames.csv').loc[:,'id':]
all_cam_2d.columns = ['oid', 'frame', 'x', 'y', 'cid']
c1_tracks = all_cam_2d[all_cam_2d['cid']=='K1'].reset_index(drop=True)
c2_tracks = all_cam_2d[all_cam_2d['cid']=='K2'].reset_index(drop=True)

# IMPORTANT - this is NOT actually x,y - but row, col
c1_2d = c1_tracks.loc[:,['x','y']].to_numpy() 
c2_2d = c2_tracks.loc[:,['x','y']].to_numpy()

#Load the dlt coefficients - and infer the Projection matrix. We already
# know the intrinsic matrix (common to all cameras)

# camera image is 640 x 512
px,py = 320, 256
fx, fy = 526, 526 # in pixels

Kteax = np.array([[fx, 0, px],
                  [0, fy, py],
                  [0, 0,  1]])

p1, p2 = np.float32([0,0]) # tangential distortion
k1, k2, k3 = np.float32([-0.3069, 0.1134, 0]) # radial distortion
dist_coefs = np.array([k1, k2, p1, p2, k3]) #in the opencv format
# Shift the origin from top left to lower left OTHERWISE THE DLT COEFS DONT MAKE SENSE
c1_2d[:,0] = py*2 - c1_2d[:,0] 
c2_2d[:,0] = py*2 - c2_2d[:,0]

c1_tracks_botleft = c1_tracks.copy()
c1_tracks_botleft['x'] = c1_2d[:,1]
c1_tracks_botleft['y'] = c1_2d[:,0]
c1_tracks_botleft['oid'] = 'k1_'+c1_tracks_botleft['oid'].astype(str) 
c1_tracks_botleft['cid'] = 1


c2_tracks_botleft = c2_tracks.copy()
c2_tracks_botleft['x'] = c2_2d[:,1]
c2_tracks_botleft['y'] = c2_2d[:,0]
c2_tracks_botleft['oid'] = 'k2_'+c2_tracks_botleft['oid'].astype(str) 
c2_tracks_botleft['cid'] = 2
# #%%
# # apply undistortion now OR NOT - it seems to complicate things for now. 
cam1_undist = cv2.undistortPoints(c1_2d, Kteax, dist_coefs, P=Kteax).reshape(-1,2)
cam2_undist = cv2.undistortPoints(c2_2d, Kteax, dist_coefs, P=Kteax).reshape(-1,2)

c1_tracks_botleft['x'] = cam1_undist[:,1]
c1_tracks_botleft['y'] = cam1_undist[:,0]
c2_tracks_botleft['x'] = cam2_undist[:,1]
c2_tracks_botleft['y'] = cam2_undist[:,0]
# # 

# plt.figure()
# plt.subplot(121)
# plt.plot(c1_2d[:,1], c1_2d[:,0],'*')
# plt.subplot(122)
# plt.plot(c2_2d[:,1], c2_2d[:,0],'*')
# # plt.plot(cam1_undist[:,1], cam1_undist[:,0],'r*')
# plt.grid()

#%% Now initialise camera objects with their projection matrices, along with 
# the F matrix that translates 2D points from one camera to the other. 
dlt_coefs = pd.read_csv('2018-08-17/2018-08-17_wand_dltCoefs_round4.csv', header=None).to_numpy()
c1_dlt, c2_dlt, c3_dlt  = [dlt_coefs[:,i] for i in range(3)]
def extract_P_from_dlt_v2(dltcoefs):
    '''No normalisation
    '''
    dltcoefs = np.append(dltcoefs, 1)
    dltcoefs = dltcoefs

    P = dltcoefs.reshape(3,4)
    return P

# generate projection matrix 
Pcam1 = extract_P_from_dlt_v2(c1_dlt)
Pcam2 = extract_P_from_dlt_v2(c2_dlt)

transl_cam1 = []
rot_cam1 = []

cam1 = Camera(0, [None, None] ,fx, px, py, fx, fy, Kteax, transl_cam1, rot_cam1, dist_coefs,
              [None]*3, Pcam1)

cam2 = Camera(0, [None, None] ,fx, px, py, fx, fy, Kteax, transl_cam1, rot_cam1, dist_coefs,
              [None]*3, Pcam2)
F = calcFundamentalMatrix(cam1, cam2)

threed_matches = match_2dpoints_to_3dtrajectories(cam1, cam2, 
                                                  c1_tracks_botleft,
                                                  c2_tracks_botleft, F, backpred_tol=200)

#%% Something's not working - why?
from track2trajectory.match2d import get_epiline

fnum = 20
k1_frame = c1_tracks_botleft[c1_tracks_botleft['frame']==fnum]
k2_frame = c2_tracks_botleft[c2_tracks_botleft['frame']==fnum]

image_k1 = plt.imread(f'ben_postfpn/K1/K1_{fnum}_mask.png')
image_k2 = plt.imread(f'ben_postfpn/K2/K2_{fnum}_mask.png')

source_point = np.array([325, 296])

r = get_epiline(k2_frame, F, cam1, cam2, source_point).flatten()

# Draw the epipolar line on K2 if cam1 is the source
# from https://www.geeksforgeeks.org/python-opencv-epipolar-geometry/
_, c, _ = image_k2.shape
x0, y0 = map(int, [0, -r[2] / r[1] ])
x1, y1 = map(int, 
             [c, -(r[2] + r[0] * c) / r[1] ])

plt.figure()
a0 = plt.subplot(121)
plt.imshow(image_k1)
plt.scatter(source_point[0],source_point[1],  s=80, facecolors='none', edgecolors='r')
plt.subplot(122, sharex=a0, sharey=a0)
plt.imshow(image_k2)
plt.plot([x0,x1], [y0,y1])






