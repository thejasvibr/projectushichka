# -*- coding: utf-8 -*-
"""
How reliable is the alignment? Part 1
=====================================
Julian Jandeleit (JJ) worked on estimating the pose and location of three 
thermal cameras in a LiDAR scan of the Orlova Chuka cave. He managed to align
the two data streams (thermal video and LiDAR), and quantified the error
using backprojection error.

Here I (Thejasvi Beleyur, TB) will exploit the fact that all microphone positions
are known through camera triangulation `and` that they are placed on the cave surface. 
If the camera-LiDAR alignment is good, then the microphones should lie on/very close to
the LiDAR scan. 

In this report I will first just visualise the results and generate a video.

Data sources
------------
1. LiDAR mesh - JJ's Bachelor thesis GitLab repo https://gitlab.inf.uni-konstanz.de/julian.jandeleit/ushichka-registration
 (Uni Konstanz access only)
1. Microphone xyz positions - TB OwnCloud repo https://owncloud.gwdg.de/index.php/apps/files/?dir=/thermo-lidar-alignment/thermal_video/2018-08-17/mic_and_wall_points/2018-08-17/mic_2D_and_3D&fileid=1596461167
1. Transformation matrix entries - JJ's Bachelor thesis (Appendix, page 18)

Package version on Windows conda environment nussl
--------------------------------------------------
1. matplotlib 3.5.1
1. pyvista 0.33.2
1. numpy 1.20.3
1. pandas 1.3.5

Author
-------
Thejasvi Beleyur, March 2022
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import pandas as pd
import pyvista as pv

#%% The transformation matrix A on page 18 of Julian's thesis (Appendix)
# This 3x4 matrix transforms homogenous coordinates from the camera to 
# the LiDAR space
A = np.array(([-0.7533, 0.6353, -0.1699, -1.7994],
              [-0.6575, -0.7332, 0.1734, 1.7945],
              [-0.0144, 0.2424, 0.9701, 0.2003]))

#%% 
# Let's load the 3D coordinates of the microphones. These xyz coordinates
# are in the camera calibration space. 

mic_xyz = pd.read_csv('data/DLTdv7_data_2018-08-17_P02_micpointsxyzpts.csv').dropna()
mic_xyz = [mic_xyz.loc[each,:].to_numpy() for each in mic_xyz.index]
mic_xyzh = [np.append(each, 1) for each in mic_xyz]

# Now move the mic from camera calibration space to LiDAR space.
mic_lidar = [ np.matmul(A, each) for each in mic_xyzh]

#%% 
# Now load the triangulated mesh which represents a sub-section of the cave - as
# cut out by Julian 

mesh = pv.read('data/lidar_roi.ply')
# cpos = mesh.plot(show_edges=True)
# print('...converting to triangulated surface...')
# surf = mesh.delaunay_3d(alpha=0.02, progress_bar=True)
# surf.save('triangulated.vtk')
# print('...DONE with triangulated surface...')

#%%
plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=True, color=True)

#%% And also add the microphones as small spheres
# in the LiDAR scan. If everything is correct -- we should see
# the wall mics ON/CLOSE to the cave wall, and the tristar sticking out with
# the obvious shape on the right side of the cave. 

mics = [pv.Sphere(radius=0.05, center=each) for each in mic_lidar]
for mic in mics:
    plotter.add_mesh(mic)

# only required if you want to re-run the video making code - to change camera
# angles and positions. Left-click on a mesh point and camera details are printed out
# def callback(x):
#     print(x)
#     print(f'camera position: {plotter.camera.position}')
#     print(f'camera az,rol,elev: {plotter.camera.azimuth},{plotter.camera.roll},\
#           {plotter.camera.elevation}')
#     print(f'camera view angle, focal point: {plotter.camera.view_angle,plotter.camera.focal_point}')


# Open a movie file
plotter.open_movie('thermo-lidar_alignment.mp4', framerate=25, quality=4)
plotter.write_frame()  # write initial data
plotter.add_text("Orlova-Chuka mic positions in cave", name='video-label', position='lower_right')

#plotter.track_click_position(callback)
#%%
start_camera_pos = (3.7123199465845342, 0.19909826300124092, -1.041653569330262)
end_camera_pos = (2.8658470324818657, -1.7164564338765609, -1.6956853891383652)

camera_az = (0, 0)
camera_roll = (-89, -100)
camera_elev = (0.5, 0.5)
fpoint_start = (-3.749681192709187, 2.752914417889359, 0.27757022381482277)
fpoint_end =   (-3.9629910869837053, 2.1443213882625427, -0.133988232544467)

steps = 70
position_transition = np.column_stack((np.linspace(start_camera_pos[0], end_camera_pos[0], steps),
                                       np.linspace(start_camera_pos[1], end_camera_pos[1], steps),
                                       np.linspace(start_camera_pos[2], end_camera_pos[2], steps)))


fpoint_start = (-3.749681192709187, 2.752914417889359, 0.27757022381482277)
fpoint_transition = np.column_stack((np.linspace(fpoint_start[0], fpoint_end[0], steps),
                                       np.linspace(fpoint_start[1], fpoint_end[1], steps),
                                       np.linspace(fpoint_start[2], fpoint_end[2], steps)))
view_angle = 30
roll_transition = np.linspace(-89, -100, steps)
#%%

plotter.camera.position = start_camera_pos
plotter.camera.azimuth = 0
plotter.camera.roll = -90
plotter.camera.elevation = 0.5 #-15

plotter.camera.view_angle = 30
plotter.camera.focal_point = (-3.749681192709187, 2.752914417889359, 0.27757022381482277)

#plotter.screenshot('camera_view1_thermo-lidar_aligned.png', window_size=(4,6))

for frame in range(steps):
    plotter.camera.position = tuple(position_transition[frame,:])
    plotter.camera.focal_point = tuple(fpoint_transition[frame,:])
    plotter.camera.roll = roll_transition[frame]
    
    plotter.render()
    plotter.write_frame()  # Write this frame

plotter.close()









