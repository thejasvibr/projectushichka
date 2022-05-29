# -*- coding: utf-8 -*-
"""
Mic alignment: 2018-07-21
=========================
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
trans_mats = glob.glob('2018-07-21/**/transf*.csv')
mic_points = glob.glob('2018-07-21/*xyzpts.csv')
mic_xyz = pd.read_csv(mic_points[0]).dropna()
mic_xyz = [mic_xyz.loc[each,:].to_numpy() for each in mic_xyz.index]
mic_xyzh = [np.append(each, 1) for each in mic_xyz]

#%%
for trans_mat in trans_mats:
    A = pd.read_csv(trans_mat, header=None).to_numpy()
    # Now move the mic from camera calibration space to LiDAR space.
    pre_post_dist, preposticp_xyz, icp_refine_transmat = run_pre_and_post_icp_steps(mic_xyzh,
                                                                                mesh, A)
    median_dists = list(map(np.median, pre_post_dist))
    range_dists = list(map(lambda X: [np.max(X), np.min(X)], pre_post_dist))
    print(median_dists, range_dists)

#%%
# Let's visualise the fit

plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=True, color=True)

mics = [pv.Sphere(radius=0.05, center=each) for each in preposticp_xyz[1]]
for mic in mics:
    plotter.add_mesh(mic)

plotter.show()

#%% 
# How far apart are the microphones when compared across the different cameras?

