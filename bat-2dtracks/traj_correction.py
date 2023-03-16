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
import napari
from skimage.io import ImageCollection
#%% Load the 2D trajectories from each of the cameras
linked_data = glob.glob('K*.csv')
cam_linked_data = {camid : pd.read_csv(linked) for camid, linked in zip(['K1','K2','K3'], linked_data) }

#%% Convert DataFrame to napari track-format
def conv_to_napari_track(df):
    track_data = []
    for i, row in df.iterrows():
        track_data.append(row.loc['id':].to_numpy())
    return track_data

#%% 
# Perform required manual corrections and then assign each trajectory a unique
# ID
k3_view = napari.Viewer()
k3_masks = ImageCollection('ben_postfpn/K3/*.png', as_gray=True)
k3_cleaned = ImageCollection('cleaned_imgs/K3/*.png', as_gray=True)
k3_tracks = conv_to_napari_track(cam_linked_data['K3'])

k3_view.add_image(
    np.array(k3_masks), 
    name="mask",
    opacity=0.9,
)

k3_view.add_image(
    np.array(k3_cleaned), 
    name="image",
    opacity=0.9,
)
k3_view.add_tracks(
    k3_tracks, 
    name="Tracks", 
    blending="translucent"
)

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



