# -*- coding: utf-8 -*-
"""
Making mesh from point cloud
============================
Let's use the open3d and pyvista packages to make the .ply point cloud
into a triangulated mesh. 

"""
import pyvista as pv

# Now load the small slice of the cave cut out by Julian to reduce computational 
# load. 
points = pv.read('data/pcroi.ply')
#cpos = mesh.plot(show_edges=True)
print('...converting to triangulated surface...')
surf = points.delaunay_3d(alpha=0.02, progress_bar=True)
surf.save('triangulated_pcroi.vtk')
print('...DONE with triangulated surface...')
