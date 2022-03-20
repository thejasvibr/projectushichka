# -*- coding: utf-8 -*-
"""
Making mesh from point cloud - open3d
=====================================
Let's use the open3d  packages to make the .ply point cloud
into a triangulated mesh. 

References
----------
1. https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba

"""
import datetime as dt
import numpy as np 
import open3d as o3d

#%%
# Now load the small slice of the cave cut out by Julian to reduce computational 
# load. 
points = o3d.io.read_point_cloud('data/pcroi.ply')

#%% Get relevant parameters for rolling ball meshing 

distances = points.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist
print(f'radius: {radius}')
#%%
print(f'...generating mesh...{dt.datetime.now()}')
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(points,
                                                                           o3d.utility.DoubleVector([radius, radius * 2]))
print(f'mesh generated...{dt.datetime.now()}')
#%% 
# Simplify mesh 
print('simplifying mesh...')
dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

dec_mesh.remove_degenerate_triangles()
dec_mesh.remove_duplicated_triangles()
dec_mesh.remove_duplicated_vertices()
dec_mesh.remove_non_manifold_edges()
print('mesh simplified...')
#%% 
print('writing mesh...')
o3d.io.write_triangle_mesh('data/triangulated_pcroi.ply', dec_mesh)
print('..mesh written.')