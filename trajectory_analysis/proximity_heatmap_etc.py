# -*- coding: utf-8 -*-
"""
Proximity profiles
==================
This analysis will explore and try to quantify various biologically relevant
movement parameters that we could extract from the 


TODO
~~~~
* Use a larger ROI and close-up the cave mesh - some rays don't ever hit the cave mesh because of the gaps in the
ceiling/floor

* Some more work needs to be done in dealing with nans in the interpolation/extrapolation
of xyz, velocities and accelerations.

DONE
~~~~
* Find a more rigorous way to model the roll in the flight trajectories. UPDATE:
  I realised what I need is not actual body roll, but really the head roll - as
  this is the 'sensor' here. Anecdotal evidence from Ofri Eitan, Arjan Boonman,
  Lasse Jakobsen and Felix Haefele suggest that bats keep their heads level - at
  least in gentle turns. In sharp turns the head likely follows along the body a bit. 
anecdotal evidence actually suggests that bats DONT roll their heads as they turn (see
email exchanges in ab.mpg.de account).
* TICK. Clean up the specks and free-floating points in the cave mesh - it affects the
ray-tracing.

Thoughts
~~~~~~~~
* Can do ray-tracing. And now what? 
* 

Created on Tue Sep 12 19:36:41 2023

@author: theja
"""


import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd
import pyvista as pv
from scipy.spatial import distance, transform, distance_matrix
from scipy.interpolate import interp1d
import sys 
sys.path.append('..\\bat-video-tracking\\')
from tracking_correction_utils import interpolate_xyz, calc_speed
import tqdm
#%%
datafolder = os.path.join('..',
                          'thermo_lidar_alignment')
mesh_path = os.path.join(datafolder, 'data','lidar_roi_holes_closed.ply')
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
rotation_angle = -60
mesh.rotate_z(rotation_angle, inplace=True)
#%%
# Rotate the mesh and all points to make for an intuitive coordinate system 

points = pv.PointSet(tmc1000_xyz)
points.rotate_z(rotation_angle, inplace=True)

# for points of 7000tmc
points7k  = pv.PointSet(tmc7000_xyz)
points7k.rotate_z(rotation_angle, inplace=True)


#%% This is commented out on purpose -  uncomment to run!!!!!
# save the rotated mesh and trajectory points
# pv.save_meshio('orlova_chuka_nice_coodsystem.ply', mesh)
tmc1000_rotated = tmc1000.copy()
tmc1000_rotated.loc[:,'x':'z'] = points.points
tmc1000_rotated.loc[:,'frame':].to_csv('2018-08-17_P01_1000TMC_200frames_nicecood_system.csv')


tmc7000_rotated = tmc7000.copy().reset_index(drop=True)
tmc7000_rotated.loc[:,'x':'z'] = points7k.points
tmc7000_rotated.loc[:,'frame':].to_csv('2018-08-17_P01_7000TMC_50frames_nicecood_system.csv')


#%%
target_dataset = tmc7000
target_rotated = tmc7000_rotated


# def keyboard_callback():
#     print(plot2.camera.position)
#     print(plot2.camera.azimuth, plot2.camera.roll, plot2.camera.elevation)
#     print(plot2.camera.view_angle)


    
# plot2 = pv.Plotter()
# plot2.add_mesh(mesh, opacity=0.3)
# plot2.camera_position = 'xz' # (0, -10, 0) #(-4.0658486275620795, 0.8678076888611709, 25) #(3.75, -2.05, -0.57)
# plot2.camera.position = (0,-5,0)
# # plot2.camera.azimuth = 0
# # plot2.camera.roll = 90
# # plot2.camera.elevation = 0 #-15

# plot2.camera.view_angle = 60
# plot2.add_points(points , color='r', render_points_as_spheres=True, point_size=10)

# axis_arrows = [pv.Arrow(np.zeros(3), each) for each in [[1,0,0], [0,1,0], [0,0,1]]]
# for c ,each in zip(['r','g','b'],axis_arrows):
#     plot2.add_mesh(each, color=c)
# plot2.add_key_event('c', keyboard_callback)
# plot2.show()

#%%
trajnums = target_rotated['id'].unique()
this_trajid = 5
thistraj = target_rotated[target_rotated['id']==trajnums[this_trajid]]
thistraj_xyz = thistraj.loc[:,'x':'z'].to_numpy()
thistrajpoints = pv.PolyData(thistraj_xyz)
#thistrajpoints.rotate_z(rotation_angle, inplace=True)

#%%

plt.figure()
for j in range(3):
    plt.plot(thistraj_xyz[:,j], label=j)
plt.legend()

#%%
multiplot = pv.Plotter()

multiplot.add_mesh(mesh, opacity=0.3)
multiplot.add_points(thistrajpoints , color='r', render_points_as_spheres=True, point_size=10)

multiplot.show()



#%% 
# How 2D is the analysis actually? 
def get_valuerange(X):
    return np.nanmax(X)-np.nanmin(X)

xyz_minmax = lambda X: X.loc[:,'x':'z'].apply(get_valuerange, axis=0)
outup  = target_dataset.groupby('id').apply(xyz_minmax)

#%% 
# Construct a proximity heatmap based on 
# One idea to build the radial heatmap is to create two spheres - one inside
# another and check all points that are within the larger sphere but not in 
# in the smaller sphere. 
# To calculate the angles (azimuth, elev), it may be 
# worth drawing a line from the bat's position to the point and projecting it 
ids = target_rotated['id'].unique()
rot_byid = target_rotated.groupby('id')

traj_index = 5

thistraj = rot_byid.get_group(ids[traj_index])
thistraj_points = thistraj.loc[:,'x':'z'].to_numpy()
#%%
# Calculate accelaration: 
# This formula is modified from Khandelwal & Hedrick 2021/2?
# TODO - implement a spline version.
from scipy.interpolate import interp1d, UnivariateSpline,InterpolatedUnivariateSpline
from scipy.interpolate import make_smoothing_spline

fps = 25
frame_durn = 1/fps

xyz_interpfun = {}

v_xyz_interpfun = {}
a_xyz_interpfun = {}

no_na_rows = np.sum(~np.isnan(thistraj_points), 1)>0
nona_thistrajpoints = thistraj_points[no_na_rows,:]
nona_t = np.arange(thistraj_points.shape[0])[no_na_rows]

xyz_interp = np.zeros(nona_thistrajpoints.shape)
v_xyz_interp = np.zeros(nona_thistrajpoints.shape)
a_xyz_interp = np.zeros(nona_thistrajpoints.shape)

for col, axis in enumerate(['x','y','z']):
    
    xyz_interpfun[axis] = make_smoothing_spline(nona_t,
                                          nona_thistrajpoints[:,col])
    xyz_interp[:,col] = xyz_interpfun[axis](nona_t)
    v_xyz_interpfun[axis] = xyz_interpfun[axis].derivative()
    v_xyz_interp[:,col] = v_xyz_interpfun[axis](nona_t)/frame_durn
    
    a_xyz_interpfun[axis] = v_xyz_interpfun[axis].derivative()
    a_xyz_interp[:,col] = a_xyz_interpfun[axis](nona_t)/frame_durn**2

rows, _ = v_xyz_interp.shape
thistraj_dirvectors = np.zeros((rows-1,3))

for i in range(1,rows):
    delta_xyz = xyz_interp[i,:]  - xyz_interp[i-1,:] 
    thistraj_dirvectors[i-1,:] = delta_xyz/np.linalg.norm(delta_xyz)

# #%%
# colnum = 2
# axes= ['x','y','z']
# plt.figure()

# plt.plot(xyz_interp[:,colnum], label=axes[colnum] + 'interpolated')
# plt.plot(thistraj_points[:,colnum], label='raw ' + axes[colnum])
# plt.plot(v_xyz_interp[:,colnum], label='interp v')
# plt.plot(a_xyz_interp[:,colnum], label='interp a')

# plt.legend()

#%%
# Try out ay/(az+9.81) - Khandelwal & Tyson Roy Soc. 
# Here the assumption is that the lizard was flying in the x direction
# 
# tan-1((ax + ay)/ (az + 9.81)) --modified version 
# Very detailed option: XYZ Cubic splines - take the derivative twice and use this
# to calculate accelaration 
# 


ax, ay, az = [a_xyz_interp[:,i] for i in range(3)]
# coarsely calculate the roll angle:
factor = 5 # IMPORTANT---THIS IS JUST SOME KIND OF SHORT-TERM FIXXXXXXX
roll_angle = np.arctan2(ax+ay, az + factor*9.81)
roll_angle_degrees = np.degrees(roll_angle)

# plt.figure()
# plt.plot(roll_angle_degrees)
# plt.legend()

#%%
from egocentric_axes import *

global_axis=[pv.Arrow([0,0,0], np.eye(3)[row,:], tip_radius=0.05, scale=0.25) for row in range(3) ]        
phi_theta , ray_points = make_rays(np.linspace(0, 330, 4) - 30, np.linspace(0, 180, 4))

movie_mode = True
traj_file = '2018-08-17_P01_7000TMC'

plot4 = pv.Plotter()
if movie_mode:
    plot4.open_movie(f'{traj_file}_{ids[traj_index]}_egocentric.mp4', framerate=fps, quality=7)
plot4.add_mesh(mesh, opacity=0.3)
plot4.add_points(xyz_interp , color='r', render_points_as_spheres=True, point_size=10)
plot4.camera_position = 'xz' # (0, -10, 0) #(-4.0658486275620795, 0.8678076888611709, 25) #(3.75, -2.05, -0.57)
plot4.camera.position = (0,-5,0)
plot4.camera.view_angle = 60

#Do  A TRANSLATE AND ROTATE OF THE RAYS https://github.com/pyvista/pyvista-support/issues/149

all_bat2cave = []
all_ego_orientations = []
#for i in tqdm.trange(thistraj_dirvectors.shape[0]):
for i in np.arange(0,thistraj_dirvectors.shape[0]):
    # # now generate the ego-centric axes considering flight direction and roll
    ego_xyz = calculate_2egocentric_xyz(thistraj_dirvectors[i,:], 0)
    all_ego_orientations.append(ego_xyz)
    #
    # What I actually need to do is to run a line from the ego-centric coordinates 
    # at various azimuth and elevation angles - and to then calculate the distance
    # between the intersection and the centre of the bat.
    rotation_mat, ssd = transform.Rotation.align_vectors(ego_xyz, np.eye(3))
    
    # throw out rays at the required angular density - centred around the origin
    bat_to_cave_distances = pd.DataFrame(columns=['frame','dist_to_surf'], index=range(ray_points.shape[0]))
    bat_to_cave_distances['frame'] = i
    bat_to_cave_distances['ray_index'] = range(ray_points.shape[0])
    # Rotate the rays to match the rotation of the egocentric system and centre them
    # on the animal
    transformed_raypoints = rotation_mat.apply(ray_points) 
    intersection_points, extended_rays = get_all_ray_intersections(transformed_raypoints,
                                                                   mesh, xyz_interp[i,:],
                                                                   10)
    bat_to_cave_distances['dist_to_surf'] = distance_matrix(xyz_interp[i,:].reshape(1,3), 
                                                            intersection_points).flatten()
    all_bat2cave.append(bat_to_cave_distances)
    header = plot4.add_text(f'Ushichka {traj_file} traj id: {ids[traj_index]}, frame: {i}')
    if i==0:
        rays = [pv.Line(xyz_interp[i,:], extended_rays[each,:]) for each in range(extended_rays.shape[0])]
        ray_actors = []
        for ray in rays:
            plot4.add_mesh(ray, )
        # initialise egocentric coordinate system 
        ego_arrows = [pv.Arrow(xyz_interp[i,:], ego_xyz[row,:], scale=0.5) for row in range(3)]
        egoaxis_actors = []
        for c ,each in zip(['r','g','b'], ego_arrows):
            egoaxis_actors.append(plot4.add_mesh(each, color=c))
        
        # intersection_actor = plot4.add_points(intersection_points, color='maroon', render_points_as_spheres=True,
        #                   point_size=30)
        intersection_spheres = []
        intersection_actors = []
        for ii, each in enumerate(intersection_points):
            intersection_spheres.append(pv.Sphere(radius=0.3, center=each))
            intersection_actors.append(plot4.add_mesh(intersection_spheres[ii], color='maroon'))
        plot4.remove_scalar_bar()
    else:
        
        for row in range(extended_rays.shape[0]):
            rays[row].points = np.row_stack((xyz_interp[i,:], 
                                             extended_rays[row]))
            displacement = intersection_points[row,:] - intersection_spheres[row].center
            intersection_spheres[row].translate(displacement, inplace=True)
            # intersection_spheres[row].translate(extended_rays[row,:]-intersection_points[row],transform_all_input_vectors=True, 
            #                                     inplace=True)
            #plot4.update_coordinates(extended_rays[row], rays[row],  render=False)
        for jj, egoarrow in enumerate(ego_arrows):
            disp = xyz_interp[i,:] - xyz_interp[i-1,:]
            ego_rotmat, ssd = transform.Rotation.align_vectors(ego_xyz[jj,:].reshape(1,3),
                                                               all_ego_orientations[i-1][jj,:].reshape(1,3))
            #transform_mat = np.eye(4)
            #transform_mat[:3,:3] = ego_rotmat.as_matrix()
            #transform_mat[:3,3] = 1
            egoarrow.translate(disp, inplace=True)
            rot_vec = ego_rotmat.as_rotvec()
            rot_radians = np.linalg.norm(rot_vec)
            rot_degrees = np.degrees(rot_radians)
            egoarrow.rotate_vector(rot_vec, rot_degrees, inplace=True, point=xyz_interp[i,:])
            
    
    if movie_mode:
        plot4.write_frame() 
    plot4.remove_actor(header)

plot4.close()
#%%

bat2cave = pd.concat(all_bat2cave).reset_index(drop=True)
#intuitive_to_phitheta = {j : dirn for j, dirn in enumerate(['left', 'back', 'down', 'right', 'front', 'up'])}

colors = plt.colormaps.get_cmap('tab20c').resampled(ray_points.shape[0]).colors

plt.figure()
for (ray_idx, subdf) in bat2cave.groupby('ray_index'):
    plt.plot(subdf.sort_values('frame')['dist_to_surf'], color=colors[ray_idx])
    #plt.plot(subdf.sort_values('frame')['dist_to_surf'], label=intuitive_to_phitheta[ray_idx])
plt.title("Traj index " + str(traj_index))
plt.legend()


#%%
# Setting up the sensory regions. 
# Front : 315-45 deg
# Right : 45 - 135 deg
# Back  : 135 - 225 
# Left  : 225 - 315 deg

phi_vals = np.linspace(0, 330, 12) - 30
theta_vals  = np.linspace(0, 180, 5)

phi_theta, rays = make_rays(phi_vals, theta_vals)

plot = pv.Plotter()
sphere = pv.Sphere(radius=1.0, )
plot.add_mesh(sphere, color='r', opacity=0.2)

all_rays = []
for direction, ray in zip(phi_theta, rays):
    all_rays.append(pv.Line([0,0,0], ray*2))

[plot.add_mesh(each, color='k') for each in all_rays]


for axis,cc in zip(np.eye(3), ['r','g','b']):
    plot.add_mesh(pv.Line([0,0,0], axis), color=cc, line_width=20)
plot.camera_position = 'xz' # 

plot.show()



