# -*- coding: utf-8 -*-
"""
Correspondence matching and 3d triangulation
============================================
Perform correspondence matching of tracks and then get 3D trajectories. 

Previously run module: traj_correction.py


Conda Environment to activate: tproject


Notes
~~~~
* This module is built from a mix of quickexample.py and thermalcamera.py in the
tproject package examples folder. 
"""
import cv2
import numpy as np
import pandas as pd
from track2trajectory.projection import project_to_2d_and_3d, calcFundamentalMatrix
from track2trajectory.match3d import match_2dpoints_to_3dtrajectories

#%% Load the camera 2D points
all_cam_2d = pd.read_csv('all_camera_tracks_2018-08-17_P01_7000_first25frames.csv').loc[:,'id':]
all_cam_2d.columns = ['oid', 'frame', 'x', 'y', 'cid']
c1_tracks = all_cam_2d[all_cam_2d['cid']=='K1'].reset_index(drop=True)
c2_tracks = all_cam_2d[all_cam_2d['cid']=='K2'].reset_index(drop=True)

c1_2d = c1_tracks.loc[:,['x','y']].to_numpy()
c2_2d = c2_tracks.loc[:,['x','y']].to_numpy()

#%% Load the dlt coefficients - and infer the Projection matrix. We already
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

# apply undistortion now
cam1_undist = cv2.undistortPoints(c1_2d, Kteax, dist_coefs, P=Kteax)
cam2_undist = cv2.undistortPoints(c2_2d, Kteax, dist_coefs, P=Kteax)

#%%







