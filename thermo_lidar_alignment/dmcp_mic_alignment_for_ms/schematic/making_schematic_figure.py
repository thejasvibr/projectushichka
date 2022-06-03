# -*- coding: utf-8 -*-
"""
Code to make thermal image and mesh view 
========================================



Created on Thu Jun  2 15:47:18 2022

@author: theja
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import pandas as pd
import numpy as np 
import pyvista as pv

image_data = pd.read_csv('2018-08-17_P01_6000_0009.csv', header=None).to_numpy()
image_data = image_data[:,:-1]

#%%
cmap = cm.coolwarm
plt.figure(figsize=(2, 2.4))
ax = plt.subplot(111)
ax.imshow(image_data, norm=colors.LogNorm(vmin=image_data.min(),vmax=image_data.max()+5),
                   cmap='seismic')
plt.yticks([]);plt.xticks([]);
plt.savefig(fname='K2_2018-08-17_small.png')

#%%
T1_pose = np.array([[      1.39,       1.83,       0.52],
         [      1.58,      -1.32,      -1.60],
         [      0.10,      -2.07,      -0.27]])
T2_pose = np.array([[      1.72,       1.87,       0.42],
         [      1.95,      -1.33,      -1.62],
         [      0.48,      -2.07,      -0.27]])
T3_pose = np.array([[      1.53,       2.19,       0.49],
                    [      1.93,      -1.03,      -1.52],
                    [      0.49,      -1.82,      -0.17]])

#%%
mesh = pv.read('../lidar/lidar_roi.ply')
#%%
camera_poses = [pv.Dodecahedron(0.1, center=T2_pose[0]),
                pv.Octahedron(0.1, center=T2_pose[1]),
                pv.Sphere(0.1, center=T2_pose[2]) ]
plotter = pv.Plotter()
# cam1 : dodecahedron, octahedron, sphere
    
for camera, col in zip(camera_poses,['red','green','blue']):
    plotter.add_mesh(camera, color=col)
plotter.add_mesh(mesh, show_edges=False, color=True)
plotter.camera.position = (3.5, -4.5, -1.5)
plotter.camera.azimuth = 5
plotter.camera.roll = -100
plotter.camera.elevation = 0 #-15
plotter.camera.view_angle = 40
plotter.camera.clipping_range = (2, 15)
plotter.save_graphic('mesh_view.pdf')
plotter.save_graphic('mesh_view.eps')
plotter.save_graphic('mesh_view.svg')
plotter.show()
#%%


