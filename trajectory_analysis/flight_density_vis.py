# -*- coding: utf-8 -*-
"""
Slicing the Orlova-Chuka mesh to get 3D bat occupation densities
================================================================

Created on Mon Aug 14 10:19:38 2023

@author: theja
"""
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd
import pyvista as pv

datafolder = os.path.join('..',
                          'thermo_lidar_alignment')
mesh_path = os.path.join(datafolder, 'data','lidar_roi.ply')
mesh = pv.read(mesh_path)
#%%
rotation_angle = -60
rotmesh = mesh.rotate_z(rotation_angle, inplace=False)
slices = mesh.slice_along_axis(n=10, axis='x',)
rotslices = rotmesh.slice_along_axis(n=10, axis='x',)
#%%
# Load bat trajectories and rotate the points to match the cave too. 
# Data is for 2018-08-17 session
A = np.array(([-0.7533, 0.6353, -0.1699, -1.7994],
              [-0.6575, -0.7332, 0.1734, 1.7945],
              [-0.0144, 0.2424, 0.9701, 0.2003]))

B = pd.read_csv(os.path.join(datafolder,'v2_transformation.csv')).to_numpy()[:,1:] # round 2 post ICP

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

bat_xyz = bat_traj.loc[:,['x','y','z']].to_numpy()
bat_xyz_lidar = []
for each in bat_xyz:
    first_t = np.matmul(A, np.append(each,1))
    xyz_lidar = np.matmul(B, np.append(first_t,1))[:-1]
    bat_xyz_lidar.append(xyz_lidar)

bat_traj_lidar = bat_traj.copy()
bat_traj_lidar.loc[:,['x','y','z']] = np.array(bat_xyz_lidar).reshape(-1,3)

bat_points = pv.PolyData(bat_traj_lidar.loc[:,"x":"z"].to_numpy()).rotate_z(rotation_angle)

#%% Also load another set of bat points from 
bat_micdata = pd.read_csv(os.path.join('..','bat-video-tracking',
                                       '2018-08-17_7000TMC_first25frames_bat-and-mic-lidarframe.csv'))

mic_row = ['m-' in each for each in bat_micdata['point_id']]
mics = bat_micdata.loc[mic_row,:]
bats = bat_micdata.loc[np.invert(mic_row),:]

new_bats = pv.PolyData(bats.loc[:,'x':'z'].to_numpy()).rotate_z(rotation_angle)


#%%
# Make many cubes stacked on top of each other. The cubes are all ~bat sized
# at 30 cm size. 
minxyz = np.min(mesh.points, axis=0)
maxxyz = np.max(mesh.points, axis=0)

plotter = pv.Plotter()
plotter.camera.position = (2.13, -5.72, -0.57)
plotter.camera.azimuth = -6
plotter.camera.roll = -98
plotter.camera.elevation = 0.5 #-15

plotter.camera.view_angle = 45

nx, ny, nz = 30, 33, 15
origin = np.array([-4,-2,-2])
grid_spacing = np.array([0.3, 0.3, 0.3])
spatial_grid = pv.ImageData(dimensions=(nx, ny, nz, nx*ny*nz),
                            spacing=grid_spacing,
                            origin=origin)

spatial_grid.cell_data["values"] = np.zeros(spatial_grid.n_cells)

# Create  a count for the number of trajectory points in the cubes.
for each in bat_points.points:
    cell_num = spatial_grid.find_closest_cell(each)
    spatial_grid.cell_data["values"][cell_num] += 1
#%%
boring_cmap = plt.cm.get_cmap("viridis", 
                              int(spatial_grid.cell_data['values'].max()))

plotter.add_volume(spatial_grid, 
                 scalars="values", cmap=boring_cmap,
                 opacity='sigmoid_6')
plotter.add_mesh(rotmesh, opacity=0.5)
plotter.add_mesh(new_bats)
plotter.show()


#%%
# Plot each of the trajectories into the mesh volume
folderpath = '..\\bat-video-tracking\\tracking_data\\2018-08-17\\'
tmc1000 = pd.read_csv(os.path.join(folderpath,'trajectories_lidarframe_2018-08-17_P01_1000TMC.csv'))
tmc7000 = pd.read_csv(os.path.join(folderpath,'trajectories_lidarframe_2018-08-17_P01_7000TMC.csv'))
tmc12000 = pd.read_csv(os.path.join(folderpath, 'trajectories_lidarframe_2018-08-17_P01_12000TMC.csv'))
tmc1000_xyz = tmc1000.loc[:,['x','y','z']].to_numpy()
tmc7000_xyz = tmc7000.loc[:,['x','y','z']].to_numpy()
tmc12000_xyz = tmc12000.loc[:,['x','y','z']].to_numpy()

plot2 = pv.Plotter()
plot2.add_mesh(mesh, opacity=0.5)
plot2.camera.position = (2.13, -5.72, -0.57)
plot2.camera.azimuth = -6
plot2.camera.roll = -98
plot2.camera.elevation = 0.5 #-15

plot2.camera.view_angle = 45
plot2.add_points(tmc1000_xyz , color='r', render_points_as_spheres=True, point_size=20)
plot2.add_points(tmc7000_xyz , color='g', render_points_as_spheres=True, point_size=20)
plot2.add_points(tmc12000_xyz , color='w', render_points_as_spheres=True, point_size=20)

plot2.show()



#%%

