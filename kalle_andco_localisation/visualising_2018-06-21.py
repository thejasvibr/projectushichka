# -*- coding: utf-8 -*-
"""
2018-06-21 SFS mic and speaker localisations - visualising the results 
======================================================================

Created on Wed Jul  2 10:21:29 2025

@author: theja
"""

import glob
import numpy as np 
import pyvista as pv
import matplotlib.pyplot as plt
import pandas as pd
import os 
import open3d as o3d
#%%

output_path = os.path.join('20250614T064950_s1_2018-06-21_good','s1_2018-06-21_11_r_sol3d.txt')

df = pd.read_csv(output_path, header=None)
df = df.T
mic_xyz_sfs = df.to_numpy()
#df.loc[:,1] *= -1 # 'flip' along the y-axis

fig3d = plt.figure()
a0 = plt.subplot(111, projection='3d')
plt.plot(df.loc[:,0], df.loc[:,1], df.loc[:,2], '*')
a0.set_aspect('equal')
plt.title('2018-06-21')
plt.show()



#%%
speaker_file = os.path.join('20250614T064950_s1_2018-06-21_good','s1_2018-06-21_12_s_sol3d.txt')
speaker_df = pd.read_csv(speaker_file, header=None).T
speaker_xyz = speaker_df.to_numpy()
#speaker_xyz[:,1] *=-1

#%%

plot2 = pv.Plotter()

mic_labels = np.arange(1, mic_xyz_sfs.shape[0]+1).tolist()
# plot2.add_points(xyz_points , color='r', render_points_as_spheres=True, point_size=10)
actor = plot2.add_point_labels(
    mic_xyz_sfs,
    mic_labels,
    italic=True,
    font_size=10,
    point_color='red',
    point_size=20,
    render_points_as_spheres=True,
    always_visible=True,
    shadow=True,
)

actor2 = plot2.add_points(speaker_xyz, color='r')

#pl.camera_position = 'xy'
plot2.add_title('2018-06-21')
plot2.show()

#%% For session 2018-06-21 - find the rotation matrix that makes channel 2 & 3 into 
# 0 height. 

mic_xyz_gt = pd.read_csv('microphone_arraygeom_2018-06-21.csv')
mic_xyz_gt = np.float64(mic_xyz_gt.loc[:,'x':'z'].to_numpy())


all_totalstation = pd.read_csv(os.path.join('C:\\',
                                              'Users',
                                              'theja',
                                              'Downloads',
                                              'mic_position_estimation',
                                              '2018-06-21',
                                              'mic2mic_measurements',
                                              'Cave_w_channel_numbers.csv'))
good_mics =  ['S0','S1','S2','S3','M1','M2','M4','M5','M6','M7']
mic_pos = all_totalstation[~pd.isna(all_totalstation['channel_num'])]
subset_micpos = mic_pos[mic_pos['Object'].isin(good_mics)]
subset_micpos = subset_micpos.sort_values(by='channel_num')
mic_xyz_gt = subset_micpos.loc[:,['X','Y','Z']].to_numpy()
#$mic_xyz_gt = np.column_stack((mic_xyz_gt[:,1], mic_xyz_gt[:,0], mic_xyz_gt[:,2]))


#%%
plot3 = pv.Plotter()

mic_labels = np.arange(1, mic_xyz_gt.shape[0]+1).tolist()
plot2.add_points(mic_xyz_gt , color='r', render_points_as_spheres=True, point_size=10)
actor1 = plot3.add_point_labels(
    mic_xyz_gt,
    mic_labels,
    italic=True,
    font_size=10,
    point_color='green',
    point_size=20,
    render_points_as_spheres=True,
    always_visible=True,
    shadow=True,
)

plot3.add_title('2018-06-21')
plot3.show()

#%%


plot4 = pv.Plotter()

sfs_mic_labels = [f'sfs {each}' for each in range(1, mic_xyz_sfs.shape[0]+1)]
actor1a = plot4.add_point_labels(
    mic_xyz_sfs,
    sfs_mic_labels,
    italic=True,
    font_size=10,
    point_color='red',
    point_size=20,
    render_points_as_spheres=True,
    always_visible=True,
    shadow=True,
)

plot4.add_title('2018-06-21')
plot4.show()


#%%
from kabsch_heidenreich_implementation import kabsch_numpy

# THE HEIDENREICH IMPLEMENTATION ALWAYS HAS SOME OFFSET - IT DOESN'T GIVE 
# PERFECT RESULTS EVEN WHENT HERE'S NO NOISE!!

P = np.random.randn(200, 3)

alpha = np.random.rand() * 2 * np.pi
R = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1]])
t = np.random.randn(3) * 10

Q = np.dot(P, R.T) + t


(rotn, translation, rmsd)  = kabsch_numpy(P, Q)


transformed = np.dot(P, rotn.T) + translation


(rotn, transl, rmsd)  = kabsch_numpy(mic_xyz_sfs, mic_xyz_gt)
sfs_mic_transf = np.dot(mic_xyz_sfs, rotn.T) + transl
# dO AN ICP ON TOP OF THE KABSCH TRANSFORM!!!

#%%

plot3b = pv.Plotter()
mic_labels = np.arange(1, mic_xyz_gt.shape[0]+1).tolist()
# plot2.add_points(xyz_points , color='r', render_points_as_spheres=True, point_size=10)
actor1 = plot3b.add_point_labels(
    mic_xyz_gt,
    mic_labels,
    italic=True,
    font_size=10,
    point_color='red',
    point_size=20,
    render_points_as_spheres=True,
    always_visible=True,
    shadow=True,
)

pt_labels = [f'sfs {each}'for each in  np.arange(1,sfs_mic_transf.shape[0]+1)]
actor2 = plot3b.add_point_labels(
    sfs_mic_transf,
    pt_labels,
    italic=True,
    font_size=10,
    point_color='green',
    point_size=20,
    render_points_as_spheres=True,
    always_visible=True,
    shadow=True,
)

plot3b.show()

