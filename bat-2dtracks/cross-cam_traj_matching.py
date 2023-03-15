# -*- coding: utf-8 -*-
"""
Cross-camera trajectory matching
================================
Attempts at manually matching trajectories from different cameras


"""
import pandas as pd 
import glob
import numpy as np 
import matplotlib
import trackpy as tp
#%% Load the 2D trajectories from each of the cameras
linked_data = glob.glob('first*.csv')
cam_linked_data = {camid : pd.read_csv(linked) for camid, linked in zip(['K1','K2','K3'], linked_data) }

#%% 
# Perform required manual corrections and then assign each trajectory a unique
# ID

def correct_trajectory_labels(linked_trajs, traj_labels, action):
    corrected = linked_trajs.copy()
    if action == 'delete':
        for each in traj_labels:
            assert len(each)==1
            linked_trajs = corrected[~corrected['particle']==each]
    if action == 'fuse':
        for each in traj_labels:
            assert len(each)==2
            relevant_rows = np.logical_or(corrected['particle']==each[0],
                                          corrected['particle']==each[1])
            corrected.loc[relevant_rows, 'particle']
            

# For K3, delete trajectories 8,7,11,16,0,5 and fuse {3,10}, {1,12}

k3_corrected = cam_linked_data['K3'].copy()
k3_traj_delete = [7,8,11,16,0,5]
k3_fuse = [(3,10), (1,12)]

#%% 
# Try to match the trajectories with each other by drawing the epipolar lines


#%% 
# Having matched trajectories across cameras - now generate the 3D tracks


#%%
# Convert bat trajectories from camera space to LiDAR space using Julian's 
# coefficients

#%% Also include the microphone positions on the cave surface. 

#%% 
# Make an animation of the bat flight trajectories with the cave and microphone array!



