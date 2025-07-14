# -*- coding: utf-8 -*-
"""
Visualising the mic array self-localisation
===========================================
Created on Sun Jun 22 23:39:24 2025

This module does all the SFS, camera-to-LiDAR and SFS-to-LiDAR transformations. 
These transformations include rotation, translation and mirroring. 




Notes
-----
* The channel order is not as expected on 2018-08-17. Channel numbers using 0-indexing
    * The notes in the Zenodo update says the order is SMP1-7, SANKEN9-12,  followed 
    by SMP8, SMP1'-8' (one-prime to eight-prime). 
    However, just looking at the inferred mic geometry - we have SMP1-6 (channel num 0-5),
    then SANKEN9-12 (tristar) on channelnums 6-9. 




@author: theja
"""
import glob
import numpy as np 
import pyvista as pv
import matplotlib.pyplot as plt
import pandas as pd
import os 
import open3d as o3d
np.random.seed(82319)
#%%

output_path = os.path.join('2025-06-18_Sv__Bat_Call_Localization__Status_meeting','*mic*.txt')
mic_files = glob.glob(output_path)

converting_session_to_date = {'session_4' : '2018-07-21',
                              'session_5B' : '2018-07-25',
                              'session_6B' : '2018-07-28',
                              'session_7B' : '2018-08-14',
                              'session_8B' : '2018-08-17',
                              'session_9B' : '2018-08-19'}
def get_session_num(filename):
    '''

    Parameters
    ----------
    filename : str
        Filename is 'kalle_results_session_<>B_mics.txt'

    Returns
    -------
    sesssion_num : str
        The session number
    '''
    parts = filename.split('_')
    session_num = parts[-4]+'_'+parts[-3]
    return session_num


j = 5
session_num = get_session_num(mic_files[j])
session_date = converting_session_to_date[session_num]

print(f'The session date is: {session_date}')

df = pd.read_csv(mic_files[j], header=None)
df = df.T
xyz_points = df.to_numpy()
#%%
# SFS derived points - may show mirroring along the wall axis


plot2a = pv.Plotter()
mic_labels = np.arange(1, xyz_points.shape[0]+1).tolist()
# plot2.add_points(xyz_points , color='r', render_points_as_spheres=True, point_size=10)
actor2a = plot2a.add_point_labels(
    xyz_points,
    mic_labels,
    italic=True,
    font_size=10,
    point_color='red',
    point_size=20,
    render_points_as_spheres=True,
    always_visible=True,
    shadow=True,
)

plot2a.add_title('SFS estimates: ' + converting_session_to_date[session_num])
plot2a.show()

#%%
speaker_file = '_'.join(mic_files[j].split('_')[:-2])+'_soundpath_.txt'
speaker_df = pd.read_csv(speaker_file, header=None).T
speaker_xyz = speaker_df.to_numpy()


#%% And now use known transformation matrix
# to align the mic points with the real-world thermal & LiDAR coordinates. 
date = session_date
micxyz_folder = os.path.join('..','thermo_lidar_alignment','dmcp_mic_alignment_for_ms')
if date == '2018-07-28':
    mic_xyz_date = pd.read_csv(os.path.join(micxyz_folder,date,
                                                    f'DLTdv7_data_{date}_mics_round2xyzpts.csv'))
elif date=='2018-08-17':
    mic_xyz_date = pd.read_csv(os.path.join(micxyz_folder,date,
                                                    f'DLTdv7_data_{date}_P02_micpointsxyzpts.csv'))
elif date=='2018-08-19':
    mic_xyz_date = pd.read_csv(os.path.join(micxyz_folder,date,
                                                    'DLTdv7_data_2081-08-19_micpositionsxyzpts.csv'))
else:
    mic_xyz_date = pd.read_csv(os.path.join(micxyz_folder,date,
                                                f'DLTdv7_data_{date}_micsxyzpts.csv'))

micxyz_camera = mic_xyz_date.dropna().to_numpy()
mic_xyzh = [np.append(each, 1) for each in micxyz_camera]

A = pd.read_csv(os.path.join(micxyz_folder,f'{date}',f'{date}--cam1','transform.csv'), 
                header=None)

if date == '2018-08-14':
    A = pd.read_csv(os.path.join(micxyz_folder,f'{date}',f'{date}--cam2','transform.csv'), 
                    header=None)

    
A = A.to_numpy()
print(A)
# Now move the mic from camera calibration space to LiDAR space.
mic_lidar = np.array([ np.matmul(A, each) for each in mic_xyzh])[:,:-1]

print(mic_lidar)
#%%
# Load cave mesh
lidar_path = os.path.join('..','thermo_lidar_alignment','data','lidar_roi_holes_closed.ply')
cavemesh = mesh = pv.read(lidar_path)

#%%
plotter = pv.Plotter(shape=(1, 2))

plotter.subplot(0,1)
mic_labels = np.arange(1, mic_lidar.shape[0]+1).tolist()
# plot2.add_points(xyz_points , color='r', render_points_as_spheres=True, point_size=10)
actor1 = plotter.add_point_labels(
    mic_lidar,
    mic_labels,
    italic=True,
    font_size=10,
    point_color='red',
    point_size=20,
    render_points_as_spheres=True,
    always_visible=True,
    shadow=True,
)

plotter.add_mesh(cavemesh, show_edges=False, color=True, opacity=0.5)
plotter.camera.position = (5.0, -1.02, -1)
plotter.camera.azimuth = 5
plotter.camera.roll = -90
plotter.camera.elevation = 5 #-15
plotter.camera.view_angle = 45

plotter.add_title(converting_session_to_date[session_num])


plotter.subplot(0,0)
mic_labels = [ f' SFS: {each}'  for each in np.arange(1, xyz_points.shape[0]+1).tolist()]
# plot2.add_points(xyz_points , color='r', render_points_as_spheres=True, point_size=10)
plotter.add_point_labels(
    xyz_points,
    mic_labels,
    italic=True,
    font_size=10,
    point_color='red',
    point_size=20,
    render_points_as_spheres=True,
    always_visible=True,
    shadow=True,
)
plotter.show()

#%% If it's the 2018-08-14 - find the 2d plane going through channels (1,2,3,4,5,6,7,12,13,14,15) (1-index)
def get_plane_equation_from_points(P, Q, R):  
    '''
    Function from 
    https://medium.com/@conghung43/get-plane-equation-from-givent-points-in-3d-coordinate-system-python-ddfccc7e0c92

    '''
    x1, y1, z1 = P
    x2, y2, z2 = Q
    x3, y3, z3 = R
    a1 = x2 - x1 
    b1 = y2 - y1 
    c1 = z2 - z1 
    a2 = x3 - x1 
    b2 = y3 - y1 
    c2 = z3 - z1 
    a = b1 * c2 - b2 * c1 
    b = a2 * c1 - a1 * c2 
    c = a1 * b2 - b1 * a2 
    d = (- a * x1 - b * y1 - c * z1) 
    return a, b, c, d


def mirror_point(a, b, c, d, x1, y1, z1): 
    '''
    Fucntion from 
    https://www.geeksforgeeks.org/dsa/mirror-of-a-point-through-a-3-d-plane/
    '''
     
    k =(-a * x1-b * y1-c * z1-d)/float((a * a + b * b + c * c))
    x2 = a * k + x1
    y2 = b * k + y1
    z2 = c * k + z1
    x3 = 2 * x2-x1
    y3 = 2 * y2-y1
    z3 = 2 * z2-z1
    return np.column_stack((x3, y3, z3))

#%%

def find_a_nonzerocoeff_plane(threedpts, max_attempts=1000):
    '''
    Parameters
    ----------
    threedpts : (N,3) np.array
        A set of 3D points lying approximately on a plane
        
    max_attempts : int, optional
        The number of sampling attempts made before giving up and raising an 
        error. This is done because the trivial solution is a,b,c,d = 0, which
        happens some times for a set of 3 points. 
        Defaults to 1000 attempts.

    Returns
    -------
    (a,b,c,d) : tuple 
        Tuple with a,b,c,d coefficients that determine a plane. 
    
    '''
    all_zeros = True
    attempts = 0
    while all_zeros:
        three_random_points = np.random.choice(np.arange(threedpts.shape[0]), 3)
        P,Q,R = [threedpts[each,:] for each in three_random_points]
        a, b, c, d = get_plane_equation_from_points(P, Q, R) 
        if not np.allclose(np.array([a,b,c,d]), np.zeros(4)):
            all_zeros = False
        else:
            attempts += 1 
            if attempts > max_attempts:
                raise ValueError(f'No compatible non-zero solutions found after {max_attempts} tries.')
    return (a,b,c,d)
    


#%%
if date=='2018-07-28':
    channel_inds = np.array([5,6,7,8]) - 1 
    xyzpts = xyz_points[channel_inds,:]
    a,b,c,d = find_a_nonzerocoeff_plane(xyzpts)
    realworld_mirrored = mirror_point(a,b,c,d, xyz_points[:,0], xyz_points[:,1], xyz_points[:,2])
    speaker_xyz_mirrored = mirror_point(a,b,c,d, speaker_xyz[:,0], speaker_xyz[:,1], speaker_xyz[:,2])
    
    plot2b = pv.Plotter()
    mic_labels = [ f' SFS mirror: {each}'  for each in np.arange(1, xyz_points.shape[0]+1).tolist()]
    
    actor2 = plot2b.add_point_labels(
        realworld_mirrored,
        mic_labels,
        italic=True,
        font_size=10,
        point_color='green',
        point_size=20,
        render_points_as_spheres=True,
        always_visible=True,
        shadow=True,
    )
    
    plot2b.show()
    
    
    
if date == '2018-08-14':
    channel_inds = np.array([1,2,3,4,5,6,7,12,13,14,15])-1
    xyzpts = xyz_points[channel_inds,:]
    # three_random_points = np.random.choice(np.arange(xyzpts.shape[0]), 3)
    # P,Q,R = [xyzpts[each,:] for each in three_random_points]
    # a, b, c, d = get_plane_equation_from_points(P, Q, R) 
    # print(a,b,c,d)
    a,b,c,d = find_a_nonzerocoeff_plane(xyzpts)
    
    
    
    realworld_mirrored = mirror_point(a,b,c,d, xyz_points[:,0], xyz_points[:,1], xyz_points[:,2])
    speaker_xyz_mirrored = mirror_point(a,b,c,d, speaker_xyz[:,0], speaker_xyz[:,1], speaker_xyz[:,2])
    
    
    


if date == '2018-08-17':
    channel_inds = np.array([1,2,4,5,6,7])-1
    xyzpts = xyz_points[channel_inds,:]
    all_zeros = True
    while all_zeros:
        three_random_points = np.random.choice(np.arange(xyzpts.shape[0]), 3)
        P,Q,R = [xyzpts[each,:] for each in three_random_points]
        a, b, c, d = get_plane_equation_from_points(P, Q, R) 
        print(a,b,c,d)
        if not np.allclose(np.array([a,b,c,d]), np.zeros(4)):
            all_zeros = False
        
        
        
    
    
    realworld_mirrored = mirror_point(a,b,c,d, xyz_points[:,0], xyz_points[:,1], xyz_points[:,2])
    speaker_xyz_mirrored = mirror_point(a,b,c,d, speaker_xyz[:,0], speaker_xyz[:,1], speaker_xyz[:,2])

    
    
    

if date == '2018-08-19':
    channel_inds = np.array([1,3,4,5,6,11,12,13,14])-1
    xyzpts = xyz_points[channel_inds,:]
    all_zeros = True
    while all_zeros:
        three_random_points = np.random.choice(np.arange(xyzpts.shape[0]), 3)
        P,Q,R = [xyzpts[each,:] for each in three_random_points]
        a, b, c, d = get_plane_equation_from_points(P, Q, R) 
        print(a,b,c,d)
        if not np.allclose(np.array([a,b,c,d]), np.zeros(4)):
            all_zeros = False
   
    
    realworld_mirrored = mirror_point(a,b,c,d, xyz_points[:,0], xyz_points[:,1], xyz_points[:,2])
    speaker_xyz_mirrored = mirror_point(a,b,c,d, speaker_xyz[:,0], speaker_xyz[:,1], speaker_xyz[:,2])
    
    
    
    
#%%
# Visualise the camera and mirrored SFS coordinates together

plotter2 = pv.Plotter(shape=(1, 2))

plotter2.subplot(0,1)
mic_labels = np.arange(1, mic_lidar.shape[0]+1).tolist()
# plot2.add_points(xyz_points , color='r', render_points_as_spheres=True, point_size=10)
actor1 = plotter2.add_point_labels(
    mic_lidar,
    mic_labels,
    italic=True,
    font_size=10,
    point_color='red',
    point_size=20,
    render_points_as_spheres=True,
    always_visible=True,
    shadow=True,
)

plotter2.add_mesh(cavemesh, show_edges=False, color=True, opacity=0.5)
plotter2.camera.position = (5.0, -1.02, -1)
plotter2.camera.azimuth = 5
plotter2.camera.roll = -90
plotter2.camera.elevation = 5 #-15
plotter2.camera.view_angle = 45

plotter2.add_title(converting_session_to_date[session_num])

try:
    print(realworld_mirrored[0,:])
except:
    realworld_mirrored = xyz_points.copy()

plotter2.subplot(0,0)
mic_labels = [ f' SFS: {each}'  for each in np.arange(1, xyz_points.shape[0]+1).tolist()]
# plot2.add_points(xyz_points , color='r', render_points_as_spheres=True, point_size=10)
plotter2.add_point_labels(
    realworld_mirrored,
    mic_labels,
    italic=True,
    font_size=10,
    point_color='red',
    point_size=20,
    render_points_as_spheres=True,
    always_visible=True,
    shadow=True,
)
plotter2.show()


#%%
# Create matching sets of points in the SFS and LiDAR coordinate systems
# set a local origin at the first mic position for the SFS & LiDAR coords
if session_date == '2018-07-21':
    channel_inds = range(4,12)
    nontristar = xyz_points[channel_inds,:]
    realworld_mirrored = xyz_points.copy()
    speaker_xyz_mirrored = speaker_xyz.copy()
    
if session_date == '2018-07-25':
    channel_inds = [8,9,10]
    nontristar = xyz_points[channel_inds,:]
    realworld_mirrored = xyz_points.copy() 
    mic_lidar_inds = [5,6,7]
    mic_lidar = mic_lidar[mic_lidar_inds,:]
    speaker_xyz_mirrored = speaker_xyz.copy()
if session_date == '2018-07-28':
    channel_inds = np.arange(5,12) - 1 
    nontristar = realworld_mirrored[channel_inds,:]
    mic_lidar_inds = np.array([1,2,4,5,6,7,8]) - 1 
    mic_lidar = mic_lidar[mic_lidar_inds,:]

if session_date == '2018-08-14':
    channel_inds = np.array([1,2,4,5,6,16,17,18]) - 1
    nontristar = realworld_mirrored[channel_inds,:]
    mic_lidar_inds = np.array([1,2,3,4,5,13,14,15]) - 1
    mic_lidar = mic_lidar[mic_lidar_inds,:]
if session_date == '2018-08-17':
    sfs_matching = np.array([1,2,4,5,7]) - 1 
    mic_lidar = mic_lidar[sfs_matching,:]
    mic_lidar_m0cent = mic_lidar - mic_lidar[0,:]
    channel_inds = sfs_matching.copy()
    nontristar = realworld_mirrored[channel_inds,:]
    speaker_xyz = speaker_xyz_mirrored

if session_date == '2018-08-19':
    channel_inds = np.array([15,16,17]) - 1 
    sfs_matching = np.array([11, 12, 13]) - 1
    mic_lidar = mic_lidar[sfs_matching,:]
    mic_lidar_m0cent = mic_lidar - mic_lidar[0,:]
    nontristar = realworld_mirrored[channel_inds,:]
    speaker_xyz = speaker_xyz_mirrored


nontristar -= nontristar[0,:]
mic_lidar_m0cent = mic_lidar - mic_lidar[0,:]
    
#%%
# align_vectors doesn't take care of the translation so that we need to do ourselves
from scipy.spatial.transform import Rotation as R
rtn, rssd, sens = R.align_vectors(mic_lidar_m0cent, nontristar, return_sensitivity=True)

sfsmics_in_lidar = rtn.apply(realworld_mirrored-realworld_mirrored[channel_inds[0],:]) + mic_lidar[0,:]
speakerxyz_wref_mic0 = speaker_xyz_mirrored - realworld_mirrored[channel_inds[0],:]
sfsspeaker_in_lidar = rtn.apply(speakerxyz_wref_mic0) + mic_lidar[0,:]
#%%

plot3b = pv.Plotter()
mic_labels = [ f' cam: {each}'  for each in np.arange(1, mic_lidar.shape[0]+1).tolist()]
# plot2.add_points(xyz_points , color='r', render_points_as_spheres=True, point_size=10)
actor1 = plot3b.add_point_labels(
    mic_lidar,
    mic_labels,
    italic=True,
    font_size=10,
    point_color='red',
    point_size=20,
    render_points_as_spheres=True,
    always_visible=True,
    shadow=True,
)

pt_labels = [f'sfs {each}'for each in  np.arange(1,sfsmics_in_lidar.shape[0]+1)]
actor2 = plot3b.add_point_labels(
    sfsmics_in_lidar,
    pt_labels,
    italic=True,
    font_size=10,
    point_color='green',
    point_size=20,
    render_points_as_spheres=True,
    always_visible=True,
    shadow=True,
)


pt_labels = [f'sfs raw {each}'for each in  np.arange(1,sfsspeaker_in_lidar.shape[0]+1)]
actor23 = plot3b.add_points(sfsspeaker_in_lidar, render_points_as_spheres=True, point_size=10.0
                            , color='grey')
poly = pv.lines_from_points(sfsspeaker_in_lidar)
plot3b.add_mesh(poly)


plot3b.add_mesh(cavemesh, show_edges=False, color=True, opacity=0.5)
plot3b.camera.position = (5.0, -1.02, -1)
plot3b.camera.azimuth = 5
plot3b.camera.roll = -90
plot3b.camera.elevation = 5 #-15
plot3b.camera.view_angle = 45

plot3b.add_title(converting_session_to_date[session_num])
plot3b.show()
