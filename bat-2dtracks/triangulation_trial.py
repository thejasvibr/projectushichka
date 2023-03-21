# -*- coding: utf-8 -*-
"""
2018-08-17_P01_7000 triangulation
=================================
Attempts at triangulating the correspondence matched points 


Previously run module
~~~~~~~~~~~~~~~~~~~~~
manual_trajmatching.py

"""
import pandas as pd
import numpy as np 

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

