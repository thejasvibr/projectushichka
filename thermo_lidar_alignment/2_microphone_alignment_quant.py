# -*- coding: utf-8 -*-
"""
How reliable is the alignment? Part 2
=====================================
Julian Jandeleit (JJ) checked the reliability of the camera-LiDAR alignment using bacprojection
errors. He reported a mean of 8 pixels backprojection error. 

 

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

References
----------
* ICP registration tutorial http://www.open3d.org/docs/latest/tutorial/Basic/icp_registration.html

Author
-------
Thejasvi Beleyur, March 2022
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import pandas as pd
import pyvista as pv
import open3d as o3d

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

#%% Now we'll proceed to calculate the nearest-neighbour distance
# between each mic point and the mesh. 

def find_closest_points_distances(points, tgt_mesh):
    
    mic_mesh_distance = []
    for micpoint in points:
        index = tgt_mesh.find_closest_point(micpoint)
        mesh_xyz = tgt_mesh.points[index]
        diff = mesh_xyz - micpoint
        distance = np.linalg.norm(diff)
        mic_mesh_distance.append(distance)
    return mic_mesh_distance

mic_mesh_distance = find_closest_points_distances(mic_lidar, mesh)
#%%
# Plot the mic-mesh distances 
# We see after transformation, the microphones are between 
# 7 mm to 34 cm off from the surface of the cave. In principle the mics
# should all be on the LiDAR surface. 

plt.figure()
plt.plot(mic_mesh_distance, 'o', markersize=10);plt.grid()
plt.ylim(0,0.5);plt.ylabel('Transformed mic-mesh distance, m', fontsize=12)
plt.yticks(fontsize=11)
plt.xticks(np.arange(len(mic_lidar)),[]);plt.xlabel('Microphones', fontsize=12)

#%% 
# Julian also suggested I try an Iterative Closest Point registration
# to improve the fit of the registration. Let's give it a try here
# 

threshold = 0.4
trans_init = np.eye(4) # give a 'blank' identity transformation matrix

mic_lidar_xyz = np.array(mic_lidar).reshape(-1,3)

source_pcd = o3d.geometry.PointCloud()
source_pcd.points = o3d.utility.Vector3dVector(mic_lidar_xyz)
target_pcd = o3d.geometry.PointCloud()
target_pcd.points = o3d.utility.Vector3dVector(np.array(mesh.points))


reg_p2p = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

transf_v2 = reg_p2p.transformation
transf_v2

pd.DataFrame(transf_v2).to_csv('v2_transformation.csv')

#%%
# Let's check the improvement after the registration again. 
miclidar_v2 = [np.matmul(transf_v2, np.append(each,1))[:-1] for each in mic_lidar]
mic_mesh_distance2 = find_closest_points_distances(miclidar_v2, mesh)

#%% Now let's compare the first and second round mic-mesh distances together. 

plt.figure(figsize=(6,4))
plt.plot(mic_mesh_distance, 'o', markersize=10, label='Julian thesis')
plt.plot(mic_mesh_distance2, 'o', markersize=10, label='Post \n ICP')
plt.grid(); plt.legend()
plt.ylim(0,0.5);plt.ylabel('Transformed mic-mesh distance, m', fontsize=12)
plt.yticks(fontsize=11)
plt.xticks(np.arange(len(mic_lidar)),[]);plt.xlabel('Microphones', fontsize=12)
plt.savefig('mic-mesh_distances.png')