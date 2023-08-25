# -*- coding: utf-8 -*-
"""
Are the XYZ positions maintained in a simplified cave mesh?
===========================================================
I made a simplified/decimated version of the Orlova Chuka cave
mesh sometime back. Now I weant to check if the  x,y,z 
coordinates of the simplified mesh are congruent with the more
complex original scan. 

Summary
-------
It does seem to be. When the microphone coordinates are transformed to 
the complex or simple meshes - the positions look very similar. 


Created on Wed Aug  9 13:30:52 2023

@author: theja
"""


import pyroomacoustics as pra
import pyvista as pv
import os 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import pandas as pd
import pyvista as pv

datafolder = '../thermo_lidar_alignment'

#%% The transformation matrix A on page 18 of Julian's thesis (Appendix)
# This 3x4 matrix transforms homogenous coordinates from the camera to 
# the LiDAR space
A = np.array(([-0.7533, 0.6353, -0.1699, -1.7994],
              [-0.6575, -0.7332, 0.1734, 1.7945],
              [-0.0144, 0.2424, 0.9701, 0.2003]))

B = pd.read_csv(os.path.join(datafolder,'v2_transformation.csv')).to_numpy()[:,1:] # round 2 post ICP
#%% 
# Let's load the 3D coordinates of the microphones. These xyz coordinates
# are in the camera calibration space. 

mic_xyz = pd.read_csv(os.path.join(datafolder,
                                   'data/DLTdv7_data_2018-08-17_P02_micpointsxyzpts.csv')).dropna()
mic_xyz = [mic_xyz.loc[each,:].to_numpy() for each in mic_xyz.index]
mic_xyzh = [np.append(each, 1) for each in mic_xyz]

# Now move the mic from camera calibration space to LiDAR space.
mic_lidar = [ np.matmul(A, each) for each in mic_xyzh]

# and include round 2 transformation 
mic_lidarv2 = [np.matmul(B, np.append(each,1))[:-1] for each in mic_lidar]

pd.DataFrame(np.row_stack(mic_lidarv2),columns=['x','y','z']).to_csv('cavealigned_2018-08-17.csv')

#%% 
# Now load the triangulated mesh which represents a sub-section of the cave - as
# cut out by Julian 

mesh = pv.read(os.path.join(datafolder,'data/lidar_roi.ply'))
simplemesh = pv.read(os.path.join('simaudio','data','smaller_slicedup_OC_evenmore.stl'))

#%%
plotter = pv.Plotter(shape=(1,2))
plotter.subplot(0,0)
plotter.add_mesh(mesh, show_edges=True, color=True)

#%% Add the microphones as small spheres
mics = [pv.Sphere(radius=0.05, center=each) for each in mic_lidarv2]
for mic in mics:
    plotter.add_mesh(mic, color='white')

#%% And load the bat trajectories and transform them to LiDAR space.
bat_traj = pd.read_csv(os.path.join(datafolder,
                                    'DLTdv8_data_p000_15000_3camxyzpts.csv'))
point_trajs = []
i = 0
for col1, col2, col3 in zip(np.arange(0,9,3), np.arange(1,9,3), np.arange(2,9,3)):
    colnames = bat_traj.columns[[col1, col2, col3]]
    new_df = bat_traj.loc[:,colnames]
    new_df.columns = ['x','y','z']
    new_df['frame'] = new_df.index
    new_df['trajId'] = str(i)
    point_trajs.append(new_df)
    i += 1

bat_traj = pd.concat(point_trajs).reset_index(drop=True)

#flightpoints = [pv.Sphere(radius=0.05, center=each) for each in bat_traj.loc[:,'x':'z'].to_numpy()]
# transofrm first to rough lidar-space
flightpoints_lidar = [ np.matmul(A, np.append(each,1)) for each in bat_traj.loc[:,'x':'z'].to_numpy()]
flightpoints_final = [ np.matmul(B, np.append(each,1))[:-1] for each in flightpoints_lidar]


flightpoints_spheres = [pv.Sphere(radius=0.05, center=each) for each in flightpoints_final]


for each in flightpoints_spheres:
    plotter.add_mesh(each, color='red')    


plotter.camera.position = (6.04, -1.02, -0.57)
plotter.camera.azimuth = -6
plotter.camera.roll = -98
plotter.camera.elevation = 0.5 #-15

plotter.camera.view_angle = 45
plotter.camera.focal_point = (0.89,-0.51,-0.25)

#%% now activate the simple mesh
plotter.subplot(0,1)
plotter.add_mesh(simplemesh)
for mic in mics:
    plotter.add_mesh(mic, color='white')
for each in flightpoints_spheres:
    plotter.add_mesh(each, color='red')  

plotter.camera.position = (6.04, -1.02, -0.57)
plotter.camera.azimuth = -6
plotter.camera.roll = -98
plotter.camera.elevation = 0.5 #-15

plotter.camera.view_angle = 45
plotter.camera.focal_point = (0.89,-0.51,-0.25)

plotter.show()


