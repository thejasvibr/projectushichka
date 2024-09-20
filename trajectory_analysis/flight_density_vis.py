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
from scipy.spatial import distance
import sys 
sys.path.append('..\\bat-video-tracking\\')
from tracking_correction_utils import interpolate_xyz
from visualisation_utils import *
#%%
datafolder = os.path.join('..',
                          'thermo_lidar_alignment')
mesh_path = os.path.join(datafolder, 'data','lidar_roi.ply')
mesh = pv.read(mesh_path)
# #%%
# rotation_angle = -60
# rotmesh = mesh.rotate_z(rotation_angle, inplace=False)
# slices = mesh.slice_along_axis(n=10, axis='x',)
# rotslices = rotmesh.slice_along_axis(n=10, axis='x',)
# #%%
# # Load bat trajectories and rotate the points to match the cave too. 
# # Data is for 2018-08-17 session
# A = np.array(([-0.7533, 0.6353, -0.1699, -1.7994],
#               [-0.6575, -0.7332, 0.1734, 1.7945],
#               [-0.0144, 0.2424, 0.9701, 0.2003]))

# B = pd.read_csv(os.path.join(datafolder,'v2_transformation.csv')).to_numpy()[:,1:] # round 2 post ICP

# bat_traj = pd.read_csv(os.path.join(datafolder,
#                                     'DLTdv8_data_p000_15000_3camxyzpts.csv'))
# point_trajs = []
# i = 0
# for col1, col2, col3 in zip(np.arange(0,9,3), np.arange(1,9,3), np.arange(2,9,3)):
#     colnames = bat_traj.columns[[col1, col2, col3]]
#     new_df = bat_traj.loc[:,colnames]
#     new_df.columns = ['x','y','z']
#     new_df['frame'] = new_df.index
#     new_df['trajId'] = str(i)
#     point_trajs.append(new_df)
#     i += 1

# bat_traj = pd.concat(point_trajs).reset_index(drop=True)

# bat_xyz = bat_traj.loc[:,['x','y','z']].to_numpy()
# bat_xyz_lidar = []
# for each in bat_xyz:
#     first_t = np.matmul(A, np.append(each,1))
#     xyz_lidar = np.matmul(B, np.append(first_t,1))[:-1]
#     bat_xyz_lidar.append(xyz_lidar)

# bat_traj_lidar = bat_traj.copy()
# bat_traj_lidar.loc[:,['x','y','z']] = np.array(bat_xyz_lidar).reshape(-1,3)

# bat_points = pv.PolyData(bat_traj_lidar.loc[:,"x":"z"].to_numpy()).rotate_z(rotation_angle)

# #%% Also load another set of bat points from 
# bat_micdata = pd.read_csv(os.path.join('..','bat-video-tracking',
#                                        '2018-08-17_7000TMC_first25frames_bat-and-mic-lidarframe.csv'))

# mic_row = ['m-' in each for each in bat_micdata['point_id']]
# mics = bat_micdata.loc[mic_row,:]
# bats = bat_micdata.loc[np.invert(mic_row),:]

# new_bats = pv.PolyData(bats.loc[:,'x':'z'].to_numpy()).rotate_z(rotation_angle)


#%%
# Make many cubes stacked on top of each other. The cubes are all ~bat sized
# # at 30 cm size. 
# minxyz = np.min(mesh.points, axis=0)
# maxxyz = np.max(mesh.points, axis=0)

# plotter = pv.Plotter()
# plotter.camera.position = (2.13, -5.72, -0.57)
# plotter.camera.azimuth = -6
# plotter.camera.roll = -98
# plotter.camera.elevation = 0.5 #-15

# plotter.camera.view_angle = 45

# nx, ny, nz = 30, 33, 15
# origin = np.array([-4,-2,-2])
# grid_spacing = np.array([0.3, 0.3, 0.3])
# spatial_grid = pv.ImageData(dimensions=(nx, ny, nz, nx*ny*nz),
#                             spacing=grid_spacing,
#                             origin=origin)

# spatial_grid.cell_data["values"] = np.zeros(spatial_grid.n_cells)

# # Create  a count for the number of trajectory points in the cubes.
# for each in bat_points.points:
#     cell_num = spatial_grid.find_closest_cell(each)
#     spatial_grid.cell_data["values"][cell_num] += 1
# #%%
# boring_cmap = plt.cm.get_cmap("viridis", 
#                               int(spatial_grid.cell_data['values'].max()))

# plotter.add_volume(spatial_grid, 
#                  scalars="values", cmap=boring_cmap,
#                  opacity='sigmoid_6')
# plotter.add_mesh(rotmesh, opacity=0.5)
# plotter.add_mesh(new_bats)
# plotter.show()


#%%
# Plot each of the trajectories into the mesh volume
folderpath = '..\\bat-video-tracking\\tracking_data\\2018-08-17\\'
tmc1000_raw = pd.read_csv(os.path.join(folderpath,'trajectories_lidarframe_2018-08-17_P01_1000TMC.csv'))
tmc7000_raw = pd.read_csv(os.path.join(folderpath,'trajectories_lidarframe_2018-08-17_P01_7000TMC.csv'))
tmc12000_raw = pd.read_csv(os.path.join(folderpath, 'trajectories_lidarframe_2018-08-17_P01_12000TMC.csv'))

interpolated_final = []
for each in [tmc1000_raw, tmc7000_raw, tmc12000_raw]:
    interpolated_dataset = []
    for batid, subdf in each.groupby('id'):
        interpolated_dataset.append(interpolate_xyz(subdf))
    interpolated_final.append(pd.concat(interpolated_dataset))


tmc1000, tmc7000, tmc12000 = interpolated_final



tmc1000_xyz = tmc1000.loc[:,['x','y','z']].to_numpy()
tmc7000_xyz = tmc7000.loc[:,['x','y','z']].to_numpy()
tmc12000_xyz = tmc12000.loc[:,['x','y','z']].to_numpy()


#%%
    
plot2 = pv.Plotter()

def keyboard_callback():
    print(plot2.camera.position)
    print(plot2.camera.azimuth, plot2.camera.roll, plot2.camera.elevation)
    print(plot2.camera.view_angle)
    
plot2.add_mesh(mesh, opacity=0.3)
plot2.camera.position = (-4.0658486275620795, 0.8678076888611709, 25) #(3.75, -2.05, -0.57)
plot2.camera.azimuth = -5
plot2.camera.roll = -66
plot2.camera.elevation = 0.5 #-15

plot2.camera.view_angle = 45
plot2.add_points(tmc1000_xyz , color='r', render_points_as_spheres=True, point_size=10)
plot2.add_points(tmc7000_xyz , color='g', render_points_as_spheres=True, point_size=10)
plot2.add_points(tmc12000_xyz , color='w', render_points_as_spheres=True, point_size=10)
plot2.add_key_event('c', keyboard_callback)


plot2.show()
#%%
#


    
plot2_1 = pv.Plotter(off_screen=False)
plot2_1.image_scale=3
def keyboard_callback():
    print(plot2_1.camera.position)
    print(plot2_1.camera.azimuth, plot2_1.camera.roll, plot2_1.camera.elevation)
    print(plot2_1.camera.view_angle)
    
plot2_1.add_mesh(mesh, opacity=0.25)
plot2_1.camera.position = (-4.0658486275620795, 0.8678076888611709, 20) #(3.75, -2.05, -0.57)
plot2_1.camera.azimuth = -5
plot2_1.camera.roll = -66
plot2_1.camera.elevation = 0.5 #-15

plot2_1.camera.view_angle = 45

for dataset  in [tmc1000, tmc7000, tmc12000]:
    for trajnum, trajdata in dataset.groupby('id'):
        xyz = trajdata.loc[:,'x':'z'].to_numpy()
        plot2_1.add_lines(xyz, width=10, connected=True, color='purple')

def callback(a, b, distance):
    plot2_1.add_text(f'Distance: {distance:.2f}', name='dist')


plot2_1.add_measurement_widget(callback)

plot2_1.show()
#plot2_1.screenshot('2018-08-17_three-trajs.png')




#%%
camera = {}
camera['position'] = (3.75, -2.05, -0.57)
camera['azimuth'] = -5
camera['roll']= -100
camera['elevation'] = 0.5 #-15
camera['view_angle'] = 45

camera2 = {}
camera2['position'] = (-4, 1, 19.22)
camera2['azimuth'] = -5
camera2['roll']= -66
camera2['elevation'] = 0.5 #-15
camera2['view_angle'] = 45


camera3 = {}
camera3['position'] = (-1.0658486275620795, 1.5, 20)
camera3['azimuth'] = -5
camera3['roll']= -69
camera3['elevation'] = 0.5 #-15
camera3['view_angle'] = 45
    

generate_incave_video(mesh, tmc1000, camera, '2018-08-17_P01-1000TMC', fps=25)
generate_incave_video(mesh, tmc1000, camera3, '2018-08-17_P01-1000TMC_top', fps=25)

generate_incave_video(mesh, tmc12000, camera, '2018-08-17_P01-12000TMC', fps=25)    
generate_incave_video(mesh, tmc12000, camera3, '2018-08-17_P01-12000TMC_top', fps=25)
#%%
from moviepy.editor import VideoFileClip, clips_array, vfx
clip1 = VideoFileClip("2018-08-17_P01-1000TMC_top.gif").margin(10) # add 10px contour
clip2 = VideoFileClip("2018-08-17_P01-1000TMC.gif").margin(10) # add 10px contour
clip3 = VideoFileClip("2018-08-17_P01-12000TMC_top.gif").margin(10) # add 10px contour
clip4 = VideoFileClip("2018-08-17_P01-12000TMC.gif").margin(10) # add 10px contour
final_clip = clips_array([[clip1, clip3],
                          [clip2, clip4]])
final_clip.resize(width=2048).write_gif("2018-08-17_P01_1000-and-12000TMC.gif", fps=25)


final_clip2 = clips_array([[clip1, clip3]])
final_clip2.resize(width=2048).write_videofile("top_2018-08-17_P01_1000-and-12000TMC.mp4")


