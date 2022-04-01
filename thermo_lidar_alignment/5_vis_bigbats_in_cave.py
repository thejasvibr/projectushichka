# -*- coding: utf-8 -*-
"""
Visualising bat flight in the cave - DETAILED
=============================================
Follow up from 4_vis_bats_in_cave.py


References
----------
1. https://track2trajectory.readthedocs.io/en/latest/




Created on Mon Mar 21 11:06:54 2022
Author: Thejasvi Beleyur, March 2022
"""
import matplotlib
from matplotlib import cm
import numpy as np 
import pandas as pd
import pyvista as pv
import tqdm
#%% The transformation matrix A on page 18 of Julian's thesis (Appendix)
# This 3x4 matrix transforms homogenous coordinates from the camera to 
# the LiDAR space
A = np.array(([-0.7533, 0.6353, -0.1699, -1.7994],
              [-0.6575, -0.7332, 0.1734, 1.7945],
              [-0.0144, 0.2424, 0.9701, 0.2003]))

B = pd.read_csv('v2_transformation.csv').to_numpy()[:,1:] # round 2 post ICP
#%% 
# Let's load the 3D coordinates of the microphones. These xyz coordinates
# are in the camera calibration space. 

mic_xyz = pd.read_csv('data/DLTdv7_data_2018-08-17_P02_micpointsxyzpts.csv').dropna()
mic_xyz = [mic_xyz.loc[each,:].to_numpy() for each in mic_xyz.index]
mic_xyzh = [np.append(each, 1) for each in mic_xyz]

# Now move the mic from camera calibration space to LiDAR space.
mic_lidar = [ np.matmul(A, each) for each in mic_xyzh]

# and include round 2 transformation 
mic_lidarv2 = [np.matmul(B, np.append(each,1))[:-1] for each in mic_lidar]

#%% 
# Now load the triangulated mesh which represents a sub-section of the cave - as
# cut out by Julian 
print('...loading mesh data...')
mesh = pv.read('data/lidar_roi.ply')
print('...mesh data loaded...')
plotter = pv.Plotter(window_size=(640, 512))
plotter.add_mesh(mesh, show_edges=False, color='tan')
# light = pv.Light()
# light.set_direction_angle(0, 0)
# plotter.add_light(light)

#%% Add the microphones as small spheres
print('..adding bat traj points..')
mics = [pv.Sphere(radius=0.05, center=each) for each in mic_lidarv2]
for mic in mics:
    plotter.add_mesh(mic, color='white')
print('..bat traj points added..')
#%% And load the bat trajectories and transform them to LiDAR space.
bat_traj = pd.read_csv('data/bat_trajs/DLTdv7_data_20180817-P000-15000xyzpts.csv')
point_trajs = []
i = 0
numcols = bat_traj.shape[1]
for col1, col2, col3 in zip(np.arange(0,numcols,3), np.arange(1,numcols,3), np.arange(2,numcols,3)):
    colnames = bat_traj.columns[[col1, col2, col3]]
    new_df = bat_traj.loc[:,colnames]
    new_df.columns = ['x','y','z']
    new_df['frame'] = new_df.index
    new_df['trajId'] = str(i)
    point_trajs.append(new_df)
    i += 1

bat_traj = pd.concat(point_trajs).reset_index(drop=True)
#%%
bat_xyz = bat_traj.loc[:,['x','y','z']].to_numpy()
bat_xyz_lidar = []
for each in bat_xyz:
    first_t = np.matmul(A, np.append(each,1))
    xyz_lidar = np.matmul(B, np.append(first_t,1))[:-1]
    bat_xyz_lidar.append(xyz_lidar)

bat_traj_lidar = bat_traj.copy()
bat_traj_lidar.loc[:,['x','y','z']] = np.array(bat_xyz_lidar).reshape(-1,3)

# bats = []
# for each in bat_xyz_lidar:
#     x,y,z = each
#     plotter.add_mesh(pv.Sphere(0.02,center=[x,y,z]), color='tan')

#%%

def callback(x):
    print(x)
    print(f'camera position: {plotter.camera.position}')
    print(f'camera az,rol,elev: {plotter.camera.azimuth},{plotter.camera.roll},\
          {plotter.camera.elevation}')
    print(f'camera view angle, focal point: {plotter.camera.view_angle,plotter.camera.focal_point}')
plotter.track_click_position(callback)


#k2_pos = (2.0402, -1.3384, -1.6065)
plotter.camera.position = (6.04, -1.02, -0.57)
plotter.camera.azimuth = -6
plotter.camera.roll = -95
plotter.camera.elevation = 0.5 #-15

plotter.camera.view_angle = 45
#plotter.camera.focal_point = (0.89,-0.51,-0.25)

# plotter.show()

#plotter.show()
#%%
# Open a movie file

plotter.open_movie('detailed_bats_in_cave.mp4', framerate=25, quality=5)
plotter.add_text("Orlova-Chuka bats in cave", name='video-label', position='lower_right')
plotter.write_frame()  # write initial data

#%%
# Begin assigning trajectory positions 

numframes = np.max(bat_traj_lidar['frame'])+1
bat_ids = bat_traj_lidar['trajId'].unique()

viridis = cm.get_cmap('viridis', len(bat_ids))
batcolors = viridis(range(len(bat_ids)))
batcolors_hex = [matplotlib.colors.to_hex(each) for each in batcolors]
bat_spheres = [pv.Sphere(0.15,center=(-999,-999,-999)) for each in bat_ids]

bat_objs = [plotter.add_mesh(each, color=batcolors_hex[i]) for i,each in enumerate(bat_spheres)]

positions = [[-999, -999, -999] for i in range(len(bat_ids))]
for frame in tqdm.trange(numframes):
    subdf = bat_traj_lidar[bat_traj_lidar['frame']==frame]
    by_bat = subdf.groupby('trajId')
    for batnum,(batid, subsubdf) in enumerate(by_bat):
        trans = subsubdf.loc[:,['x','y','z']] - np.array(positions[batnum])
        trans = tuple(trans.loc[:,['x','y','z']].to_numpy())
        if np.any(np.isnan(trans)):
            trans = (0,0,0)
            bat_spheres[batnum].translate(trans, inplace=True)
            #print(f'batid: {batid}, frame: {frame}')
        else:
            bat_spheres[batnum].translate(trans, inplace=True)
            positions[batnum] = subsubdf.loc[:,['x','y','z']]
    plotter.add_text(f"Time: {frame/25}", name='frame counter',
                      position='upper_left', color='red', font_size=12)
    plotter.add_text("Ushichka 2018-08-17 P000 15000 TMC", name='video-label', position='upper_right',
                     font_size=14)
    plotter.add_text('Ushichka homepage: https://thejasvibr.github.io/ushichka/', name='url-link', position='lower_left',
                     font_size=14, color='white')
    new_position = np.array(plotter.camera.position) + np.array([0.0015, 0.0015, 0.002])
    plotter.camera.position = tuple(new_position)

    plotter.render()
    plotter.write_frame()  # Write this frame

plotter.close()



