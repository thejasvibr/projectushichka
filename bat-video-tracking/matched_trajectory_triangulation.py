# -*- coding: utf-8 -*-
"""
2018-08-17_P01_7000 triangulation
=================================

XYZ for bat trajectories
~~~~~~~~~~~~~~~~~~~~~~~~
Attempts at triangulating the correspondence matched points. The flow is 
less than perfect - there are still some points that aren't assigned to the 
correct trajectory - but this can be handled in the next round of code alterations.

XYZ for microphone points 
~~~~~~~~~~~~~~~~~~~~~~~~~
Here we'll also get the xyz points for the microphone array.


Previously run module
~~~~~~~~~~~~~~~~~~~~~
3_manual_trajmatching_2018-08-17_7000TMC-first25f.py

"""
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv
import numpy as np 
from scipy.spatial import distance_matrix
#%%
cam_data = pd.read_csv('matched_2018-08-17_P01_7000_first25frames.csv')
dlt_coefs = pd.read_csv('2018-08-17/2018-08-17_wand_dltCoefs_round4.csv', header=None).to_numpy()
c1_dlt, c2_dlt, c3_dlt  = [dlt_coefs[:,i] for i in range(3)]

# keep only points that are matched across at least 2 cameras 
twonans_in_string = lambda X : X.count('nan')>=2
cam_data['geq_2cams'] = ~cam_data['point_id'].apply(twonans_in_string)
valid_points = cam_data[cam_data['geq_2cams']]

#%% Effect of undistortion
#   ======================
# Let's also try undistorting the 2D pixel data and see how much of an overall
# difference it makes
raw_xy = valid_points.loc[:,['x','y']].to_numpy()
import cv2

px,py = 320, 256 # x,y image centers
fx, fy = 526, 526 # in pixels

Kteax = np.array([[fx, 0, px],
                  [0, fy, py],
                  [0, 0,  1]])

p1, p2 = np.float32([0,0]) # tangential distortion
k1, k2, k3 = np.float32([-0.3069, 0.1134, 0]) # radial distortion
dist_coefs = np.array([k1, k2, p1, p2, k3]) #in the opencv format

undist_xy = cv2.undistortPoints(raw_xy, Kteax, dist_coefs, P=Kteax).reshape(-1,2)
validpoints_undistort = valid_points.copy()
validpoints_undistort.loc[:,['x','y']] = undist_xy

#%%
# Reorder the data into a DLTdv7 compatible format. 


# Refer to http://www.kwon3d.com/theory/dlt/dlt.html#3d to see how to 
# get 3D coordinates from m cameras given the DLT coefficients and x,y pixels
# are known. 

def calc_xyz_from_uv(cam_data, dlt_coefs):
    '''
    Parameters
    ----------
    cam_data : (Ncams, 3) np.array
        With columns representing [camera id, x, y]
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
    if ncams<2:
        raise ValueError(f'Only {ncams} detected. >=2 cameras required!')
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

def triangulate_multicamera_data(allcam_data, dltcoefs):
    '''
    Parameters
    ----------
    allcam_data : pd.DataFrame
        With at least columns : ['point_id', 'frame', 'cid', 'x', 'y']
    dltcoefs : (11,N) np.array
        11 DLT coefficients per camera
    
    Returns
    -------
    triangulated : pd.DataFrame
        With columns 'framenum', 'point_id', 'x', 'y', 'z'

    '''
    xdata = [] 
    ydata = []
    zdata = []
    framenums = []
    all_point_id = []
    by_id = allcam_data.groupby('point_id')
    for point_id, sub_df in by_id:
        by_frame = sub_df.groupby('frame')
        for fnum, subsub_df in by_frame:
            #print(f'frame: {fnum}')
            cam_data = subsub_df.dropna().loc[:,['cid','x','y']].to_numpy()
            if cam_data.shape[0]>1:
                soln, _ = calc_xyz_from_uv(cam_data, dltcoefs)
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
    return triangulated

triangulated  = triangulate_multicamera_data(valid_points, dlt_coefs)
triangulated_undist = triangulate_multicamera_data(validpoints_undistort, dlt_coefs)
undistorting_effect = distance_matrix(triangulated.loc[:,'x':'z'].to_numpy(),
                                      triangulated_undist.loc[:,'x':'z'].to_numpy())

#%%
# Plot the data and see if it all makes sense
fig, a9 = plt.subplots()
a9 = plt.subplot(111, projection='3d')
for pointid, sub_df in triangulated_undist.groupby('point_id'):
    plt.plot(sub_df['x'], sub_df['y'], sub_df['z'], '*')
    last_xyz = [sub_df.loc[sub_df.index[0],axis] for axis in ['x','y','z']]
    text_colo = a9.lines[-1].get_c()
    a9.text(last_xyz[0],last_xyz[1],last_xyz[2], pointid, fontsize=10, color=text_colo)
#%%
plt.figure()
plt.subplot(111)
for pointid, sub_df in triangulated_undist.groupby('point_id'):
    xyz = sub_df.loc[:,['framenum','x','y','z']]
    xyz['t'] = xyz['framenum']*0.04
    delta_txyz = np.diff(xyz.loc[:,['t','x','y','z']].to_numpy(), axis=0)
    speed = np.sqrt(np.sum(delta_txyz[:,1:]**2,axis=1))/delta_txyz[:,0]
    plt.plot(xyz['framenum'][:-1],speed, label=pointid)
plt.legend()
#%% Convert the dataframe into a numpy array for matrix manipulations 
# Assign numeric keycodes to the various point-ids
point_xyz = triangulated_undist.loc[:,['x','y','z']].to_numpy().T
point_xyz = np.row_stack((point_xyz, np.tile(1, point_xyz.shape[1])))
numeric_ids, pointid_nums = pd.factorize(triangulated['point_id'])
object_cmap_object= plt.cm.get_cmap("viridis", len(np.unique(numeric_ids)))
object_cmaps = object_cmap_object.colors


#%% XYZ for microphone points
#   ~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's also load the microphone xy points digitised with DLTdv7 and then 
# get the 3D points from it. 

# the order of digitsation
mic_names = ['1','2','3','4', '5', '6', '7', '8', '1p', '2p', '3p', '4p',
            '5p', '6p', '7p', '8p', '9', '10', '11', '12']
mic_xy = pd.read_csv('2018-08-17/DLTdv7_data_2018-08-17_P02_micpointsxypts.csv')
mic_xy_nona = mic_xy.dropna(thresh=4)
mic_xy_nona['frame'] = mic_xy_nona.index+1
mic_xy_nona['point_id'] = mic_names

allcam_micxy= []
for camnum in [1,2,3]:
    onecam_df = mic_xy_nona.loc[:,f'pt1_cam{camnum}_X':f'pt1_cam{camnum}_Y']
    onecam_df['cid'] = camnum
    onecam_df['frame'] = mic_xy_nona['frame']
    onecam_df['point_id'] = mic_xy_nona['point_id']
    onecam_df.columns = ['x', 'y', 'cid', 'frame', 'point_id']
    allcam_micxy.append(onecam_df)

allcams_mic_xy = pd.concat(allcam_micxy).reset_index(drop=True)
mic_xyz = triangulate_multicamera_data(allcams_mic_xy, dlt_coefs)
mic2mic_xyz = distance_matrix(mic_xyz.loc[:,'x':'z'], mic_xyz.loc[:,'x':'z'])
micarray  = mic_xyz.loc[[19,1,2,3],'x':'z'].to_numpy()
raw_tristar_center_toall = distance_matrix(micarray, micarray)
# and now perform undistortion and check if the tristar mic distances are closer
# to the expected. 

ideal_distmat = np.array([[0, 1.2, 1.2, 1.2],
                          [1.2, 0, 2, 2 ],
                          [1.2, 2, 0, 2],
                          [1.2, 2, 2, 0]])

raw_micxy = allcams_mic_xy.loc[:,'x':'y'].to_numpy()
undistorted_micxy = cv2.undistortPoints(raw_micxy, Kteax, dist_coefs, P=Kteax).reshape(-1,2)
allcams_undist_micxy = allcams_mic_xy.copy()
allcams_undist_micxy.loc[:,'x':'y'] = undistorted_micxy
micxyz_undist = triangulate_multicamera_data(allcams_undist_micxy, dlt_coefs)
micarray_undist  = micxyz_undist.loc[[19,1,2,3],'x':'z'].to_numpy()
undist_tristar_center_toall = distance_matrix(micarray_undist, micarray_undist)
mic2mic_undist_xyz = distance_matrix(micxyz_undist.loc[:,'x':'z'], micxyz_undist.loc[:,'x':'z'])

#%% Now align the points to the cave LiDAR system and see if the trajectories make sense
# contextually. We can then go back and make corrections. 
lidar_data = pv.read('../thermo_lidar_alignment/data/lidar_roi.ply')

# This is the transformation matrix for 2018-08-17 from Julian's thesis. 
A = np.array(([-0.7533, 0.6353, -0.1699, -1.7994],
              [-0.6575, -0.7332, 0.1734, 1.7945],
              [-0.0144, 0.2424, 0.9701, 0.2003]))
# Bring the 3d points in camera frame to LiDAR frame. 
points_lidarframe = np.apply_along_axis(lambda X: np.matmul(A, X), 0, point_xyz).T
framenums = triangulated_undist['framenum']

# move the mic xyz to the LiDAR coordinate system 
micxyz_homog =  np.column_stack((micxyz_undist.loc[:,'x':'z'].to_numpy(),
                                 np.tile(1,micxyz_undist.shape[0])))
micxyz_undist_lidarframe = np.apply_along_axis(lambda X: np.matmul(A, X), 0, micxyz_homog.T).T

# SAve the bat trajectories and microphone xyz in the lidar frame into the same csv file.
micxyz_lidarframe = micxyz_undist.copy()
micxyz_lidarframe.loc[:,'x':'z'] = micxyz_undist_lidarframe 
micxyz_lidarframe['point_id'] = 'm-'+micxyz_lidarframe['point_id']
battrajs_lidarframe = triangulated_undist.copy()
battrajs_lidarframe.loc[:,'x':'z'] = points_lidarframe
bat_and_mic = pd.concat((micxyz_lidarframe, battrajs_lidarframe)).reset_index(drop=True)
bat_and_mic.to_csv('2018-08-17_7000TMC_first25frames_bat-and-mic-lidarframe.csv')

#%%
plotter = pv.Plotter()

def callback(x):
    print(x)
    print(f'camera position: {plotter.camera.position}')
    print(f'camera az,rol,elev: {plotter.camera.azimuth},{plotter.camera.roll},\
          {plotter.camera.elevation}')
    print(f'camera view angle, focal point: {plotter.camera.view_angle,plotter.camera.focal_point}')
plotter.track_click_position(callback)

plotter.add_mesh(lidar_data, color='brown', )
camera_pos = (8.11, -5.82, -1.11)
camera_pos = (6.11, -4.2, -0.84)

plotter.camera.position = camera_pos
plotter.camera.azimuth = 0
plotter.camera.roll = -94
plotter.camera.elevation = -0 #-15
plotter.camera.view_angle = 45


for col, each, frame in zip(numeric_ids, points_lidarframe, framenums):
    plotter.add_mesh(pv.Sphere(radius=0.05, center=each),
                         color=object_cmaps[col][:-1])

# Also add the microphones into the scene
for xyz in micxyz_undist_lidarframe:
    plotter.add_mesh(pv.Sphere(radius=0.02, center=xyz),
                         color='r')
plotter.add_text('Ushichka 2018-08-17 7000 TMC 0-1 second', position='lower_left')

light = pv.Light(position=camera_pos, color='white', light_type='scene light',
                 intensity=0.3)
plotter.add_light(light)
legend_entries = []
for point_num, point_id in enumerate(pointid_nums):
    legend_entries.append([point_id, object_cmaps[point_num][:-1]])

_ = plotter.add_legend(legend_entries)
plotter.save_graphic('Ushichka-2018-08-17_P01_7000TMC_0-1second.pdf')
if __name__ == "__main__":
    plotter.show()
else:
    plotter.close()


#%%
# TODO
# ~~~~
# Make an animation together that shows the dynamic trajectories over time.