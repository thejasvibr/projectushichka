# -*- coding: utf-8 -*-
"""
Cross-camera trajectory matching
================================
Attempts at manually matching trajectories from different cameras


"""
import pandas as pd 
import glob
import matplotlib
import trackpy as tp
#%% Load the 2D trajectories from each of the cameras
linked_data = glob.glob('first*.csv')
cam_linked_data = {camid : pd.read_csv(linked) for camid, linked in zip(['K1','K2','K3'], linked_data) }

#%% 
# Perform required manual corrections and then assign each trajectory a unique
# ID


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



