# -*- coding: utf-8 -*-
"""
Proximity profiles
==================
This analysis will explore and try to quantify various biologically relevant
movement parameters that we could extract from the 

Created on Tue Sep 12 19:36:41 2023

@author: theja
"""


import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd
import pyvista as pv
from scipy.spatial import distance, transform
from scipy.interpolate import interp1d
import sys 
sys.path.append('..\\bat-video-tracking\\')
from tracking_correction_utils import interpolate_xyz, calc_speed

#%%
datafolder = os.path.join('..',
                          'thermo_lidar_alignment')
mesh_path = os.path.join(datafolder, 'data','lidar_roi.ply')
mesh = pv.read(mesh_path)


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
# Rotate the mesh and all points to make for an intuitive coordinate system 
rotation_angle = -60
mesh.rotate_z(rotation_angle, inplace=True)
points = pv.PointSet(tmc1000_xyz)
points.rotate_z(rotation_angle, inplace=True)

#%% This is commented out on purpose -  uncomment to run!!!!!
# save the rotated mesh and trajectory points
# pv.save_meshio('orlova_chuka_nice_coodsystem.ply', mesh)
# tmc1000_rotated = tmc1000.copy()
# tmc1000_rotated.loc[:,'x':'z'] = points.points
# tmc1000_rotated.loc[:,'frame':].to_csv('2018-08-17_P01_1000TMC_200frames_nicecood_system.csv')
#%%
target_dataset = tmc7000_xyz

plot2 = pv.Plotter()

def keyboard_callback():
    print(plot2.camera.position)
    print(plot2.camera.azimuth, plot2.camera.roll, plot2.camera.elevation)
    print(plot2.camera.view_angle)


    
plot2.add_mesh(mesh, opacity=0.3)
plot2.camera_position = 'xz' # (0, -10, 0) #(-4.0658486275620795, 0.8678076888611709, 25) #(3.75, -2.05, -0.57)
plot2.camera.position = (0,-5,0)
# plot2.camera.azimuth = 0
# plot2.camera.roll = 90
# plot2.camera.elevation = 0 #-15

plot2.camera.view_angle = 60
plot2.add_points(points , color='r', render_points_as_spheres=True, point_size=10)

axis_arrows = [pv.Arrow(np.zeros(3), each) for each in [[1,0,0], [0,1,0], [0,0,1]]]
for c ,each in zip(['r','g','b'],axis_arrows):
    plot2.add_mesh(each, color=c)
plot2.add_key_event('c', keyboard_callback)
plot2.show()

#%%
trajnums = tmc1000['id'].unique()
this_trajid = 1
thistraj = tmc1000[tmc1000['id']==trajnums[this_trajid]]
thistraj_xyz = thistraj.loc[:,'x':'z'].to_numpy()
thistrajpoints = pv.PolyData(thistraj_xyz)
thistrajpoints.rotate_z(rotation_angle, inplace=True)


multiplot = pv.Plotter()

median_z = np.nanmedian(thistraj['z'])

multiplot.add_mesh(mesh, opacity=0.3)
multiplot.add_points(thistrajpoints , color='r', render_points_as_spheres=True, point_size=10)
projected = thistrajpoints.project_points_to_plane(origin=[0,0,median_z])
multiplot.add_mesh(projected, color='g')
multiplot.show()

#%% 
# How 2D is the analysis actually? 
def get_valuerange(X):
    return np.nanmax(X)-np.nanmin(X)

xyz_minmax = lambda X: X.loc[:,'x':'z'].apply(get_valuerange, axis=0)
outup  = tmc1000.groupby('id').apply(xyz_minmax)

#%% 
# Construct a proximity heatmap based on 
# One idea to build the radial heatmap is to create two spheres - one inside
# another and check all points that are within the larger sphere but not in 
# in the smaller sphere. 
# To calculate the angles (azimuth, elev), it may be 
# worth drawing a line from the bat's position to the point and projecting it 
ids = tmc1000_rotated['id'].unique()
rot_byid = tmc1000_rotated.groupby('id')

traj_index = 0

thistraj = rot_byid.get_group(ids[traj_index])
thistraj_points = thistraj.loc[:,'x':'z'].to_numpy()
rows, _ = thistraj_points.shape
thistraj_dirvectors = np.zeros((rows-1,3))

for i in range(1,rows):
    delta_xyz = thistraj_points[i,:]  - thistraj_points[i-1,:] 
    thistraj_dirvectors[i-1,:] = delta_xyz/np.linalg.norm(delta_xyz)
# Set the [1,0,0] as the 'x' vector and get the corresponding orthogonal Z vector



#%%
# Calculate accelaration: 
# This formula is modified from Khandelwal & Hedrick 2021/2?
# TODO - implement a spline version.

fps = 25
frame_durn = 1/fps
v_xyz = np.apply_along_axis(np.diff, 0, thistraj_points)/frame_durn
a_xyz = np.apply_along_axis(np.diff, 0, v_xyz)/frame_durn
ax, ay, az = [abs(a_xyz[:,i]) for i in range(3)]
# coarsely calculate the roll angle:
factor = 1 # IMPORTANT
roll_angle = np.arctan2(ax+ay,az + factor*9.81)
roll_angle_degrees = np.degrees(roll_angle)

plt.figure()
plt.plot(roll_angle_degrees)
plt.legend()
#%%
# Try out ay/(az+9.81) - Khandelwal & Tyson Roy Soc. 
# Here the assumption is that the lizard was flying in the x direction
# 
# tan-1((ax + ay)/ (az + 9.81)) --modified version 
# Very detailed option: XYZ Cubic splines - take the derivative twice and use this
# to calculate accelaration 
# 

i = 5

plot3 = pv.Plotter()
    
plot3.add_mesh(mesh, opacity=0.3)
plot3.camera_position = 'xz' # (0, -10, 0) #(-4.0658486275620795, 0.8678076888611709, 25) #(3.75, -2.05, -0.57)
plot3.camera.position = (0,-5,0)

plot3.camera.view_angle = 60
plot3.add_points(thistraj_points , color='r', render_points_as_spheres=True, point_size=10)

axis_arrows = [pv.Arrow(np.zeros(3), each) for each in [[1,0,0], [0,1,0], [0,0,1]]]
for c ,each in zip(['r','g','b'],axis_arrows):
    plot3.add_mesh(each, color=c)

egocentric_axes = np.array([[1,0,0],
                            [0,1,0],
                            [0,0,1]])
# align y-axis to flight direction 
align_to_y, err = transform.Rotation.align_vectors(thistraj_dirvectors[i,:].reshape(1,3),
                                              np.array([0,1,0]).reshape(1,3))
aligned_yaxis = align_to_y.apply([0,1,0])
# Also rotate x-axis to this level
# rotated_xaxis = align_to_y.apply([1,0,0])
# rotated_zaxis = align_to_y.apply([0,0,1])
rotated_xyz = align_to_y.apply(np.eye(3))

align_to_roll = transform.Rotation.from_euler('y', -roll_angle[i-2],)

# now fix the roll by estimating the required rotations around the y-axis
# if we 

plot3.add_mesh(pv.Arrow(thistraj_points[i,:], rotated_xyz[0,:]),'r')
plot3.add_mesh(pv.Arrow(thistraj_points[i,:], rotated_xyz[1,:]),'g')
plot3.add_mesh(pv.Arrow(thistraj_points[i,:], rotated_xyz[2,:]),'b')



# THE PROPER WAY FINALLY -  NOT REALLY
# First find the flight direction and set that to the y-axis 
# The global +Z is the gravity vector and get the 'non-egocentric' x-axis to 
# the orthogonal between the +Z and flight direction. 
# Egocentric +Z is orthogonal to the non-ego X and ego Y axis. 
# Rotate all egocentric axes about the y-axis by roll angle. 

nonego_x = np.cross(np.array([0,0,-1]), 
                    aligned_yaxis)
ego_z = np.cross(nonego_x, aligned_yaxis)
nonrot_xyz = np.row_stack((nonego_x, aligned_yaxis, ego_z))
rot_xyz = align_to_roll.apply(nonrot_xyz)


#
# fORGT about getting the coordinate system perfectly aligned. 
# Where do the bats keep the cave wals (and each other) when they're flying
#spheres = [pv.Sphere(radius=r, center=thistraj_points[i,:]) for r in radius]
plot3.show()

#%%

from scipy.spatial import KDTree




# the direction of the sphere is wrong for now - but yes...
sensory_range = pv.Sphere(radius=0.15, direction=ego_z, 
                          center=thistraj_points[i,:], theta_resolution=30, 
                          phi_resolution=30)
tree = KDTree(mesh.points)
d_kdtree, idx = tree.query(sensory_range.points)
sensory_range["distances"] = d_kdtree

print(np.median(d_kdtree))
