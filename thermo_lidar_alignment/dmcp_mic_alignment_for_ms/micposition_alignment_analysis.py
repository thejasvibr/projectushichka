# -*- coding: utf-8 -*-
"""
Mic alignment: thermal-LiDAR
============================

Some thoughts:
    * Why even do ICP? In general, the challenge is really to get the coarse
    alignment right in the first place.
    * It seems like the 3D alignment errors and 2D reprojection errors
    are rather correlated - does this make sense?
    * ICP helps in some cases, but when the transformation is already off
    it only puts a tiny band-aid on a bleed :p by pushing all points to 
    any nearby mesh points.


@author: Thejasvi Beleyur
Code released under MIT License
"""
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import pandas as pd
import pyvista as pv
from common_funcs import find_closest_points_distances, icp_register
from common_funcs import run_pre_and_post_icp_steps

#%%
# First load the triangulated mesh which represents a sub-section of the cave
mesh = pv.read('lidar/lidar_roi.ply')
#%%
trans_mats = glob.glob('2018-08-18/**/transf*.csv')
mic_points = glob.glob('2018-08-18/*xyzpts.csv')
mic_xyz = pd.read_csv(mic_points[0]).dropna()
mic_xyz = [mic_xyz.loc[each,:].to_numpy() for each in mic_xyz.index]
mic_xyzh = [np.append(each, 1) for each in mic_xyz]

#%%
all_mic_pos_pre = []
all_mic_pos_post = []
for trans_mat in trans_mats:
    A = pd.read_csv(trans_mat, header=None).to_numpy()
    # Now move the mic from camera calibration space to LiDAR space.
    pre_post_dist, preposticp_xyz, icp_refine_transmat = run_pre_and_post_icp_steps(mic_xyzh,
                                                                                mesh, A,
                                                                                max_distance=1.5)
    median_dists = list(map(np.median, pre_post_dist))
    range_dists = list(map(lambda X: [np.max(X), np.min(X)], pre_post_dist))
    all_mic_pos_pre.append(preposticp_xyz[0])
    all_mic_pos_post.append(preposticp_xyz[1])
    print(median_dists, '\n', range_dists)

#%%
# Let's visualise the fit

plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=True, color=True)

mics = [pv.Sphere(radius=0.1, center=each) for each in all_mic_pos_post[1]]
for mic in mics:
    plotter.add_mesh(mic)

plotter.show()

#%% 
# How far apart are the microphones when compared across the different cameras?

