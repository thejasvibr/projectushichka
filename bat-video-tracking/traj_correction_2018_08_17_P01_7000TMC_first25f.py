# -*- coding: utf-8 -*-
"""
Cross-camera trajectory matching
================================
Attempts at manually matching trajectories from different cameras



Previously run module : btrack_checkingout.py
Module to be run after this: 

"""
import argparse
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
linked_data = glob.glob('*tracks_first*.csv')
cam_linked_data = {camid : pd.read_csv(linked).loc[:,'id':] for camid, linked in zip(['K1','K2','K3'], linked_data) }

#%% Convert DataFrame to napari track-format
def conv_to_napari_track(df):
    track_data = []
    for i, row in df.iterrows():
        track_data.append(row.loc['id':].to_numpy())
    return track_data

def replace_or_add_row(main_df, new_data):
    '''
    
    Parameters
    ----------
    main_df, new_data: pd.DataFrame
        Must have the following columns: row, col, id, frame

    Returns
    -------
    main_df : pd.DataFrame
        The same input df object, but with replaced data or 
        additional rows added in
   
    '''
    for ind, row_data in new_data.iterrows():
        row, col, particle_id, frame = [row_data[each] for each in ['row', 'col', 'id', 'frame']]
        rows_exist = np.logical_and(main_df['id']==particle_id,
                                           main_df['frame']==frame)
        if sum(rows_exist)>0:
            main_df.loc[rows_exist, 'row'] = row
            main_df.loc[rows_exist, 'col'] = col
        else:
            # now create the new rows
            new_row = main_df.index.max()+1
            main_df.loc[new_row,'id'] = particle_id
            main_df.loc[new_row,'frame'] = frame
            main_df.loc[new_row,'row'] = row
            main_df.loc[new_row,'col'] = col
    return main_df


#%% 
# K3 corrections
# ~~~~~~~~~~~~~~
# Perform required manual corrections and then assign each trajectory a unique
# ID
k3_detections = pd.read_csv('K3_2018-08-17_7000_00-2D-detections.csv').groupby('frame')

k3_view = napari.Viewer()
k3_masks = np.array(ImageCollection('cleaned_and_masks/masks/K3/*.png'))[:25]
k3_cleaned = np.array(ImageCollection('cleaned_and_masks/cleaned/K3/*.png'))[:25]
k3_tracks = conv_to_napari_track(cam_linked_data['K3'])

k3_view.add_image(
    np.array(k3_masks), 
    name="mask",
    opacity=0.6,
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

#%% Object id 5: between frame 0-60
corrected_k3 = cam_linked_data['K3'].copy().loc[:,'id':]
obj_5_centroids = pd.DataFrame([[5, 0, 95, 32],
                                [5, 1, 113, 90.5],
                                [5, 2, 128, 132],
                                [5, 3, 124, 165],
                                [5, 4, 133, 195],
                                ], columns=['id','frame','row','col'])
# particle 11 is actually also 7, and include another row in there
p_11_rows = corrected_k3['id']==11
corrected_k3.loc[p_11_rows,'id'] = 7
f21_addn = pd.DataFrame(data={'id':[7], 'frame':[21], 'row':[295], 'col':[434]})
corrected_k3 = pd.concat((corrected_k3, f21_addn)).reset_index(drop=True)
# Also  keep only these trajectory IDs - removing 12 
K3_keep_trajs = [1, 4, 5, 6, 7, 9, 10]
corrected_k3 = corrected_k3[corrected_k3['id'].isin(K3_keep_trajs)]


replace_or_add_row(corrected_k3, obj_5_centroids)

# also replace OR add rows to corrected_k3 
# for ind, row_data in obj_5_centroids.iterrows():
#     particle_id, frame, row, col = row_data
#     rows_exist = np.logical_and(corrected_k3['id']==particle_id,
#                                        corrected_k3['frame']==frame)
#     if sum(rows_exist)>0:
#         corrected_k3.loc[rows_exist, 'row'] = row
#         corrected_k3.loc[rows_exist, 'col'] = col
#     else:
#         # now create the new rows
#         new_row = corrected_k3.index.max()+1
#         corrected_k3.loc[new_row,'id'] = particle_id
#         corrected_k3.loc[new_row,'frame'] = frame
#         corrected_k3.loc[new_row,'row'] = row
#         corrected_k3.loc[new_row,'col'] = col

#%% 
# View the corrected K3 tracks

k3_corr_view = napari.Viewer()
k3_tracks = conv_to_napari_track(corrected_k3)

k3_corr_view.add_image(
    np.array(k3_masks), 
    name="mask",
    opacity=0.2,
)

k3_corr_view.add_image(
    np.invert(k3_cleaned), 
    name="image",
    opacity=0.9,
    colormap="gray_r"
)
k3_corr_view.add_tracks(
    conv_to_napari_track(corrected_k3), 
    name="Tracks", 
    blending="translucent"
)

#%% K2 corrections
#   ~~~~~~~~~~~~~~

k2_view = napari.Viewer()
k2_masks = np.array(ImageCollection('cleaned_and_masks/masks/K2/*.png'))[:25]
k2_cleaned = np.array(ImageCollection('cleaned_and_masks/cleaned/K2/*.png'))[:25]
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
    colormap='gray_r'
)
k2_view.add_tracks(
    k2_tracks, 
    name="Tracks", 
    blending="translucent"
)

#%% 
# Mostly Okay K2 tracking aside from the 
corrected_k2 = cam_linked_data['K2'].copy().loc[:,'id':]
k2_detections = pd.read_csv('K2_2018-08-17_7000_00-2D-detections.csv').groupby('frame')
# rename particle 8 to particle 3
corrected_k2.loc[corrected_k2['id'] == 8, 'id'] = 3
p6_addition = pd.DataFrame({'id':[6], 'row':[312], 'col':[480], 'frame': [23]})
p3_addition = pd.DataFrame({'id':[3,3,3,3],
                            'frame': [5,6,13,14],
                            'row':[349,348,353,353],
                            'col':[53,49,9,2],
                            })
all_additions = pd.concat((p6_addition, p3_addition))

replace_or_add_row(corrected_k2, all_additions)
# remove id 12 detection on frames 22-23
f23_p12 = np.logical_and(np.logical_or(corrected_k2['frame']==23, corrected_k2['frame']==22),
                         corrected_k2['id']==12)
corrected_k2 = corrected_k2.loc[~f23_p12,:]
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
    colormap='gray_r'
)
k2_corr_view.add_tracks(
    conv_to_napari_track(corrected_k2), 
    name="Tracks", 
    blending="translucent"
)

#%% K1 corrections 
#   ~~~~~~~~~~~~~~


k1_view = napari.Viewer()
k1_masks = np.array(ImageCollection('cleaned_and_masks/masks/K1/*.png'))[:25]
k1_cleaned = np.array(ImageCollection('cleaned_and_masks/cleaned/K1/*.png'))[:25]
k1_tracks = conv_to_napari_track(cam_linked_data['K1'])

k1_view.add_image(
    np.array(k1_masks), 
    name="mask",
    opacity=0.5,
)

k1_view.add_image(
    np.invert(k1_cleaned), 
    name="image",
    opacity=0.9,
    colormap="gray_r"
)
k1_view.add_tracks(
    k1_tracks, 
    name="Tracks", 
    blending="translucent"
)

#%% 
# First fix the 'main' bat that flies through the scene and is poorly tracked. 
k1_detections = pd.read_csv('K1_2018-08-17_7000_00-2D-detections.csv').groupby('frame')
corrected_k1 = cam_linked_data['K1'].copy().loc[:,'id':]

ids_to_fuse = [9,10,12,13]
corrected_k1.loc[corrected_k1['id'].isin(ids_to_fuse),'id'] = 8
p8_additions = pd.DataFrame(data={'id':   [8,   8,   8, 8, 8, 8],
                                  'frame':[16,  17, 20, 21, 22, 23],
                                  'row':  [205, 219, 256, 264, 275, 284],
                                  'col':  [113, 168, 313, 349, 382, 407]})
p2_additions = {'id'   :[2]*5,
                'frame':[17, 18, 19, 20, 21],
                'row':  [460, 459, 458, 454, 453,],
                'col':  [20 ,  16, 11,  7  ,  3]}
p2_additions = pd.DataFrame(data=p2_additions)

p11_addition = pd.DataFrame(data={'id':[11],
                                  'frame':[17],
                                  'row': [460],
                                  'col': [3]})

replace_or_add_row(corrected_k1, p8_additions)
replace_or_add_row(corrected_k1, p2_additions)
replace_or_add_row(corrected_k1, p11_addition)

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
    colormap="gray_r"
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
all_corrected_tracks = all_corrected_tracks.loc[:,['id','frame','row','col','camera']]
all_corrected_tracks.to_csv('all_camera_tracks_2018-08-17_P01_7000_first25frames.csv')



