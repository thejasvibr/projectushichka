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
import natsort
from skimage.io import ImageCollection
from skimage.measure import regionprops
import matplotlib.pyplot as plt 
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
k3_masks = [np.loadtxt(each, dtype='int8') for each in natsort.natsorted(glob.glob('ben_postfpn/K3/*.csv'))]
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
frames = range(7)
fig, a0 = plt.subplots()
frame_centroids = {}
for idx  in frames:
    plt.sca(a0)
    plt.imshow(k3_masks[idx])
    import scipy.ndimage as ndi
    labeled, num_areas = ndi.label(k3_masks[idx])
    object_locs = ndi.find_objects(labeled)
    mms = regionprops(labeled)
    frame_centroids[idx] = [each.centroid for each in mms]
    for each in mms:
        plt.plot(each.centroid[1], each.centroid[0], '+')
        plt.text(each.centroid[1]+10, each.centroid[0]+10, f'{int(each.centroid[0])},{int(each.centroid[1])}')
    plt.savefig(f'K3_{idx}_centroids.png')
    a0.cla()
    
corrected_k3 = cam_linked_data['K3'].copy().loc[:,'id':]
#%% Object id 21: between frame 0-60
 
obj_21_centroids = pd.DataFrame([[21, 0, 95, 31],
                                [21, 1, 95, 31],
                                [21, 2, 112, 90],
                                [21, 3, 127, 129],
                                [21, 4, 127, 162],
                                [21, 5, 130, 192],
                                [21, 6, 135, 216]], columns=['id','frame','x','y'])

#%% 
# Having matched trajectories across cameras - now generate the 3D tracks


#%%
# Convert bat trajectories from camera space to LiDAR space using Julian's 
# coefficients

#%% Also include the microphone positions on the cave surface. 

#%% 
# Make an animation of the bat flight trajectories with the cave and microphone array!



