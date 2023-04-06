# -*- coding: utf-8 -*-
"""
Cross-camera trajectory matching
================================
Attempts at manually matching trajectories from different cameras



Previously run module : btrack_checkingout.py
Module to be run after this: 

"""
import pandas as pd 
import glob
import numpy as np 
import matplotlib
import napari
import natsort
from skimage.io import ImageCollection
from skimage.measure import regionprops
import scipy.ndimage as ndi
import matplotlib.pyplot as plt 
import tqdm
#%% Load the 2D trajectories from each of the cameras
linked_data = glob.glob('K*tracks_first*.csv')
cam_linked_data = {camid : pd.read_csv(linked) for camid, linked in zip(['K1','K2','K3'], linked_data) }

#%% Convert DataFrame to napari track-format
def conv_to_napari_track(df):
    track_data = []
    for i, row in df.iterrows():
        track_data.append(row.loc['id':].to_numpy())
    return track_data

#%% 
# K3 corrections
# ~~~~~~~~~~~~~~
# Perform required manual corrections and then assign each trajectory a unique
# ID
k3_view = napari.Viewer()
k3_masks = [np.loadtxt(each, dtype='int8') for each in natsort.natsorted(glob.glob('ben_postfpn/K3/*.csv'))]
k3_cleaned = np.array(ImageCollection('cleaned_imgs/K3/*.png')[:25])
k3_tracks = conv_to_napari_track(cam_linked_data['K3'])

k3_view.add_image(
    np.array(k3_masks), 
    name="mask",
    opacity=0.2,
)

k3_view.add_image(
    np.invert(k3_cleaned), 
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
for idx  in tqdm.tqdm(frames):
    plt.sca(a0)
    plt.imshow(k3_masks[idx])
    labeled, num_areas = ndi.label(k3_masks[idx])
    object_locs = ndi.find_objects(labeled)
    mms = regionprops(labeled)
    frame_centroids[idx] = [each.centroid for each in mms]
    for each in mms:
        plt.plot(each.centroid[1], each.centroid[0], '+')
        plt.text(each.centroid[1]+10, each.centroid[0]+10, f'{int(each.centroid[0])},{int(each.centroid[1])}')
    plt.savefig(f'K3_{idx}_centroids.png')
    a0.cla()
plt.close()    

#%% Object id 21: between frame 0-60
corrected_k3 = cam_linked_data['K3'].copy().loc[:,'id':]
obj_21_centroids = pd.DataFrame([[21, 0, 95, 31],
                                [21, 1, 95, 31],
                                [21, 2, 112, 90],
                                [21, 3, 127, 129],
                                [21, 4, 127, 162],
                                [21, 5, 130, 192],
                                [21, 6, 135, 216]], columns=['id','frame','x','y'])
# Also  keep only these trajectory IDs - removing all else. 
K3_keep_trajs = [21, 36, 41, 4, 43, 5, 44]
corrected_k3 = corrected_k3[corrected_k3['id'].isin(K3_keep_trajs)]
particle21_relevant = np.logical_and(corrected_k3['id']==21, corrected_k3['frame']<7)
corrected_k3.loc[particle21_relevant,'id':] = obj_21_centroids.values

#%% 
# Having matched trajectories across cameras - now generate the 3D tracks

k3_corr_view = napari.Viewer()
k3_masks = [np.loadtxt(each, dtype='int8') for each in natsort.natsorted(glob.glob('ben_postfpn/K3/*.csv'))]
k3_cleaned = np.array(ImageCollection('cleaned_imgs/K3/*.png')[:25])[:,:,:,0]
k3_tracks = conv_to_napari_track(cam_linked_data['K3'])

k3_corr_view.add_image(
    np.array(k3_masks), 
    name="mask",
    opacity=0.2,
)

k3_corr_view.add_image(
    np.invert(k3_cleaned), 
    name="image",
    opacity=0.9,
)
k3_corr_view.add_tracks(
    conv_to_napari_track(corrected_k3), 
    name="Tracks", 
    blending="translucent"
)

#%% K2 corrections
#   ~~~~~~~~~~~~~~
k2_view = napari.Viewer()
k2_masks = [np.loadtxt(each, dtype='int8') for each in natsort.natsorted(glob.glob('ben_postfpn/K2/*.csv'))]
k2_cleaned = np.array(ImageCollection('cleaned_imgs/K2/*.png')[:25])[:,:,:,0]
k2_tracks = conv_to_napari_track(cam_linked_data['K2'])

k2_view.add_image(
    np.array(k2_masks), 
    name="mask",
    opacity=0.2,
)

k2_view.add_image(
    np.invert(k2_cleaned), 
    name="image",
    opacity=0.9,
)
k2_view.add_tracks(
    k2_tracks, 
    name="Tracks", 
    blending="translucent"
)

#%% 
# 1) Remove IDs 7, 9, 13
# 2) Remove particle 16 from frame #22, 23 (0 indexed)
k2_to_remove = [7,9,13,14]
corrected_k2 = cam_linked_data['K2'].copy()
corrected_k2 = corrected_k2[~corrected_k2['id'].isin(k2_to_remove)]
p16_todelete = np.logical_and(corrected_k2['frame']<24, corrected_k2['id']==16)
corrected_k2 = corrected_k2[~p16_todelete]

idx = 24
labeled, num_areas = ndi.label(k2_masks[idx])
object_locs = ndi.find_objects(labeled)
mms = regionprops(labeled)
frame_centroids = [each.centroid for each in mms]

#%% Visualise the corrected K2 tracks
k2_corr_view = napari.Viewer()
k2_corr_view.add_image(
    np.array(k2_masks), 
    name="mask",
    opacity=0.2,
)

k2_corr_view.add_image(
    np.invert(k2_cleaned), 
    name="image",
    opacity=0.9,
)
k2_corr_view.add_tracks(
    conv_to_napari_track(corrected_k2), 
    name="Tracks", 
    blending="translucent"
)

#%% K1 corrections 
#   ~~~~~~~~~~~~~~


k1_view = napari.Viewer()
k1_masks = [np.loadtxt(each, dtype='int8') for each in natsort.natsorted(glob.glob('ben_postfpn/K1/*.csv'))]
k1_cleaned = np.array(ImageCollection('cleaned_imgs/K1/*.png')[:25])[:,:,:,0]
k1_tracks = conv_to_napari_track(cam_linked_data['K1'])

k1_view.add_image(
    np.array(k1_masks), 
    name="mask",
    opacity=0.2,
)

k1_view.add_image(
    np.invert(k1_cleaned), 
    name="image",
    opacity=0.9,
)
k1_view.add_tracks(
    k1_tracks, 
    name="Tracks", 
    blending="translucent"
)

#%% 
# * IDS to fuse into one : [28, 30, 33, 35, 36, 37, 42] --> into (new) ID 45
# * IDs to keep: [1, 2, 6, 26, 27, 19, 34, (new)45]
# IDs 6,  34 needs a bit of manual tracking help
# (new ID) 42 may need some manual entry in missing frames. 
corrected_k1 = cam_linked_data['K1'].copy()
k1_ids_tofuse_to42 = [28,30,33,35,36,37,42]
corrected_k1.loc[corrected_k1['id'].isin(k1_ids_tofuse_to42),'id'] = 45
k1_ids_tokeep = [1, 2, 6, 26, 27, 19, 34, 45]
corrected_k1 = corrected_k1[corrected_k1['id'].isin(k1_ids_tokeep)]
id6_frame1 = corrected_k1[np.logical_and(corrected_k1['id']==6, corrected_k1['frame']==0)].index
id6_frame18on = pd.DataFrame(data={'id':[6,6,6],
                                   'frame':[18,19,20],
                                   'x':[458,458,454],
                                   'y':[16, 11, 8]})
# remove a repeat 45 detection 
repeat45_row = corrected_k1[((corrected_k1['frame']==19) & (corrected_k1['id']==45) & (corrected_k1['x']<240))]
# row to remove has repeat id45 x <= 230
corrected_k1.drop(repeat45_row.index, inplace=True)

# include 45 centroids for frame 21-23

idx = 23
labeled, num_areas = ndi.label(k1_masks[idx])
object_locs = ndi.find_objects(labeled)
mms = regionprops(labeled)
frame_centroids = [each.centroid for each in mms]

id45_frame21on = pd.DataFrame(data={'id':np.tile(45,3),
                                   'frame':[21,22,23],
                                   'x':[264, 274, 283],
                                   'y':[347, 379, 407]})


corrected_k1 = pd.concat([corrected_k1,id6_frame18on,id45_frame21on]).reset_index(drop=True).loc[:, 'id':]

#%%

k1_corr_view = napari.Viewer()
k1_corr_view.add_image(
    np.array(k1_masks), 
    name="mask",
    opacity=0.2,
)

k1_corr_view.add_image(
    np.invert(k1_cleaned), 
    name="image",
    opacity=0.9,
)
k1_corr_view.add_tracks(
    conv_to_napari_track(corrected_k1), 
    name="Tracks", 
    blending="translucent"
)

#%% SAve all corrected data into one big csv file
corrected_k1['camera'] = 'K1'
corrected_k2['camera'] = 'K2'
corrected_k3['camera'] = 'K3'

all_corrected_tracks = pd.concat([corrected_k1, corrected_k2, corrected_k3]).reset_index(drop=True)
all_corrected_tracks = all_corrected_tracks.loc[:,['id','frame','x','y','camera']]
all_corrected_tracks.to_csv('all_camera_tracks_2018-08-17_P01_7000_first25frames.csv')



