# -*- coding: utf-8 -*-
"""
2018-08-17_P01_7000 triangulation
=================================
Attempts at triangulating the correspondence matched points 


Previously run module
~~~~~~~~~~~~~~~~~~~~~
manual_trajmatching.py

"""
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv
import numpy as np 
#%%
cam_data = pd.read_csv('matched_2018-08-17_P01_7000_first25frames.csv')
dlt_coefs = pd.read_csv('2018-08-17/2018-08-17_wand_dltCoefs_round4.csv', header=None).to_numpy()
c1_dlt, c2_dlt, c3_dlt  = [dlt_coefs[:,i] for i in range(3)]

# keep only points that are matched across at least 2 cameras 
twonans_in_string = lambda X : X.count('nan')>=2
cam_data['geq_2cams'] = ~cam_data['point_id'].apply(twonans_in_string)
valid_points = cam_data[cam_data['geq_2cams']]
#%%
# Reorder the data into a DLTdv7 compatible format. 


# Refer to http://www.kwon3d.com/theory/dlt/dlt.html#3d to see how to 
# get 3D coordinates from m cameras given the DLT coefficients and x,y pixels
# are known. 

def calc_xyz_from_uv(cam_data, dlt_coefs):
    '''
    Parameters
    ----------
    cam_data : pd.DataFrame
        (Ncams, 3) xy data of the same point in one frame. 
        With columns cid, x, y
    dlt_coefs : (11,Ncams) np.array
        11 DLT coefficients  for each camera
    Returns 
    -------
    soln : (3,1) np.array
        xyz position of object
    residual : (1,) np.array
        residual from the least squares solution estimate

    See  http://www.kwon3d.com/theory/dlt/dlt.html#3d eqn. 22
    '''
    ncams = cam_data.shape[0]
    cam_row_stack_LHS = []
    cam_row_stack_RHS = []
    for i in range(ncams):
        cam_id = int(cam_data[i,0])
        L = dlt_coefs[:,cam_id-1]
        v, w = cam_data[i,[1,2]]
        lhs_camstack = np.zeros((2,3))
        rhs_camstack = np.zeros((2,1))
        lhs_camstack[0,:] = [v*L[8]-L[0], v*L[9]-L[1], v*L[10]-L[2]]
        lhs_camstack[1,:] = [w*L[8]-L[4], w*L[9]-L[5], w*L[10]-L[6]]
        rhs_camstack[:,0] = [L[3]-v, L[7]-w]
        cam_row_stack_LHS.append(lhs_camstack)
        cam_row_stack_RHS.append(rhs_camstack)
    LHS = np.row_stack(cam_row_stack_LHS)
    RHS = np.row_stack(cam_row_stack_RHS)
    soln, residual, *other = np.linalg.lstsq(LHS, RHS)
    return soln, residual 
#%%
xdata = [] 
ydata = []
zdata = []
framenums = []
all_point_id = []
by_id= valid_points.groupby('point_id')
for point_id, sub_df in by_id:
    by_frame = sub_df.groupby('frame')
    for fnum, subsub_df in by_frame:
        cam_data = subsub_df.loc[:,['cid','x','y']].to_numpy()
        soln, _ = calc_xyz_from_uv(cam_data, dlt_coefs)
        x,y,z = soln.flatten()
        xdata.append(x)        
        ydata.append(y)
        zdata.append(z)
        framenums.append(fnum)
        all_point_id.append(point_id)
triangulated = pd.DataFrame(data={'framenum': framenums,
                                  'point_id': all_point_id,
                                  'x': xdata,
                                  'y': ydata,
                                  'z': zdata})

#%%
# Plot the data and see if it all makes sense
fig, a9 = plt.subplots()
a9 = plt.subplot(111, projection='3d')
for pointid, sub_df in triangulated.groupby('point_id'):
    plt.plot(sub_df['x'], sub_df['y'], sub_df['z'], '*')
    last_xyz = [sub_df.loc[sub_df.index[0],axis] for axis in ['x','y','z']]
    text_colo = a9.lines[-1].get_c()
    a9.text(last_xyz[0],last_xyz[1],last_xyz[2], pointid, fontsize=5, color=text_colo)

#%% Convert the dataframe into a numpy array for matrix manipulations 
# Assign numeric keycodes to the various point-ids
point_xyz = triangulated.loc[:,['x','y','z']].to_numpy().T
point_xyz = np.row_stack((point_xyz, np.tile(1, point_xyz.shape[1])))
numeric_ids, _ = pd.factorize(triangulated['point_id'])

#%% Now align the points to the cave LiDAR system and see if the trajectories make sense
# contextually. We can then go back and make corrections. 
lidar_data = pv.read('../thermo_lidar_alignment/data/lidar_roi.ply')


# This is the transformation matrix for 2018-08-17 from Julian's thesis. 
A = np.array(([-0.7533, 0.6353, -0.1699, -1.7994],
              [-0.6575, -0.7332, 0.1734, 1.7945],
              [-0.0144, 0.2424, 0.9701, 0.2003]))
# Bring the 3d points in camera frame to LiDAR frame. 
points_lidarframe = np.apply_along_axis(lambda X: np.matmul(A, X), 0, point_xyz).T

plotter = pv.Plotter()
plotter.add_mesh(lidar_data, show_edges=True, color=True)
camera_pos = (2.86, -1.71, -1.69)

plotter.camera.position = camera_pos
for each in points_lidarframe:
    plotter.add_mesh(pv.Sphere(radius=0.05, center=each))
    

plotter.show()



