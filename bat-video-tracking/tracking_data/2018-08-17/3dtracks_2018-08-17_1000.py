# -*- coding: utf-8 -*-
"""
Track matching to get 3d points 2018-08-17 P01-1000 TMC
==========================================

Track generation
~~~~~~~~~~~~~~~~
TrackMate (Fiji Plugin) was used to generate and correct 2D tracks. 
The raw frames were first passed through an entropy filter with a 7 pixel
disk radius in scikit-image. 


Steps to get succesful 3D tracks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Run TrackMate on Entropy image masks of thermal data frames
* Run mask based tracking
* Perform Linear Assignment Problem (LAP) tracking 
* Manually correct the tracking results using the TrackScheme
* Save output XML file and tracks to CSV file
* Load tracks from CSV file and check once again for repeated assignments of the 
  same point
* Run track matching and generate trajectories
* Check track matching by projecting the trajectories back onto all cameras to
  see if there are any tracks/detections missing.

Conda env
~~~~~~~~~
Activate the part2ushichka environment for this module. 

Coodinate systems
~~~~~~~~~~~~~~~~~
Each package has a different 2D coordinate system output/expectation. 

DLT based functions : origin at bottom-left. Y increases in upward direction
opencv2 : origin at top-left. Y increases in downward direction 


Created on Sat Aug 26 08:14:10 2023

@author: theja
"""
import glob
from itertools import product
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import natsort
from track2trajectory.dlt_to_world import partialdlt, transformation_matrix_from_dlt, cam_centre_from_dlt
from track2trajectory.dlt_to_world import dlt_reconstruct_v2, dlt_inverse
from track2trajectory.dlt_based_matching import find_best_matches, row_wise_dlt_reconstruct
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix as distmat
from scipy.spatial import distance
import cv2
#%%
camids = ['K1', 'K2', 'K3']
trackfiles = [glob.glob(os.path.join('P01_1000TMC','linked_tracks',each+'*.csv'))[0] for each in camids]
camera_tracks = {each: pd.read_csv(filepaths, header=0,skiprows=[1,2,3]) for each, filepaths in zip(camids, trackfiles)}
for camid, df in camera_tracks.items():
    df['cam_id'] = camid

# get only a subset of columns for easier handling and rename 
trackdata = []
for camid, df in camera_tracks.items():
    trackdf = df.copy()
    trackdf = trackdf.loc[:,['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'FRAME', 'cam_id']]
    trackdf = trackdf.rename({'TRACK_ID': 'id',
                              'POSITION_X':'x',
                              'POSITION_Y': 'y',
                              'FRAME': 'frame',
                              'cam_id': 'cid'}, axis='columns').sort_values(['id','frame'])
    trackdf['cid'] = int(trackdf['cid'][0][1])
    trackdf['camera'] = 'K'+str(trackdf['cid'][0])
    trackdata.append(trackdf)

all_cam_2d = pd.concat(trackdata).reset_index(drop=True)
#%%
# Camera intrinsics
# camera image is 640 x 512
px,py = 320, 256 # x,y image centers
fx, fy = 526, 526 # in pixels

Kteax = np.array([[fx, 0, px],
                  [0, fy, py],
                  [0, 0,  1]])

p1, p2 = np.float32([0,0]) # tangential distortion
k1, k2, k3 = np.float32([-0.3069, 0.1134, 0]) # radial distortion
dist_coefs = np.array([k1, k2, p1, p2, k3]) #in the opencv format

dlt_coefs = pd.read_csv('..\\..\\2018-08-17\\2018-08-17_wand_dltCoefs_round4.csv', header=None).to_numpy()

P1 = np.append(dlt_coefs[:,0], [1]).reshape(3,4)
P2 = np.append(dlt_coefs[:,1], [1]).reshape(3,4)
P3 = np.append(dlt_coefs[:,2], [1]).reshape(3,4)

allcam_xy = all_cam_2d.loc[:,['x','y']].to_numpy()
allcam_undistxy = cv2.undistortPoints(allcam_xy, Kteax, dist_coefs,P=Kteax).reshape(-1,2)
all_cam_2d.loc[:,['x','y']] = allcam_undistxy

#%%
c1_tracks = all_cam_2d[all_cam_2d['camera']=='K1'].reset_index(drop=True)
c2_tracks = all_cam_2d[all_cam_2d['camera']=='K2'].reset_index(drop=True)
c3_tracks = all_cam_2d[all_cam_2d['camera']=='K3'].reset_index(drop=True)

c1_tracks['id'] = c1_tracks['id'].astype(int)
c2_tracks['id'] = c2_tracks['id'].astype(int)
c3_tracks['id'] = c3_tracks['id'].astype(int)


c1_2d = c1_tracks.loc[:,['x','y']].to_numpy()
c2_2d = c2_tracks.loc[:,['x','y']].to_numpy()
c3_2d = c3_tracks.loc[:,['x','y']].to_numpy()
#%%
# Shift the origin from top left to lower left OTHERWISE THE DLT COEFS DONT MAKE SENSE
c1_2d[:,1] = py*2 - c1_2d[:,1] 
c2_2d[:,1] = py*2 - c2_2d[:,1]
c3_2d[:,1] = py*2 - c3_2d[:,1]

c1_tracks_botleft = c1_tracks.loc[:,['id','frame']].copy()
c1_tracks_botleft['x'] = c1_2d[:,0]
c1_tracks_botleft['y'] = c1_2d[:,1]
c1_tracks_botleft['id'] = 'k1_'+c1_tracks_botleft['id'].astype(str) 
c1_tracks_botleft['cid'] = 1

c2_tracks_botleft = c2_tracks.loc[:,['id','frame']].copy()
c2_tracks_botleft['x'] = c2_2d[:,0]
c2_tracks_botleft['y'] = c2_2d[:,1]
c2_tracks_botleft['id'] = 'k2_'+c2_tracks_botleft['id'].astype(str) 
c2_tracks_botleft['cid'] = 2

c3_tracks_botleft = c3_tracks.loc[:,['id','frame']].copy()
c3_tracks_botleft['x'] = c3_2d[:,0]
c3_tracks_botleft['y'] = c3_2d[:,1]
c3_tracks_botleft['id'] = 'k3_'+c3_tracks_botleft['id'].astype(str) 
c3_tracks_botleft['cid'] = 3

#%%
# Perform the corrections here itself initially 


# Check for repeats in ID within a frame. If there are any - print a notification
for camdata in [c1_tracks_botleft, c2_tracks_botleft, c3_tracks_botleft]:
    for frame, framedata in camdata.groupby('frame'):
        uniqids, counts = np.unique(framedata['id'], return_counts=True)
        repeat_inds = np.where(counts>1)[0]
        if repeat_inds.size>0:
            print(framedata['cid'].unique(), frame, uniqids[repeat_inds])



#%% Now initialise camera objects with their projection matrices, along with 
# the F matrix that translates 2D points from one camera to the other. 
c1_dlt, c2_dlt, c3_dlt  = [dlt_coefs[:,i] for i in range(3)]

#%%

global fnum, k1_images, k2_images, k3_images
k1_images = natsort.natsorted(glob.glob('P01_1000TMC/cleaned/K1/*.png'))
k2_images = natsort.natsorted(glob.glob('P01_1000TMC/cleaned/K2/*.png'))
k3_images = natsort.natsorted(glob.glob('P01_1000TMC/cleaned/K3/*.png'))

fnum = 0
k1_frame = c1_tracks_botleft[c1_tracks_botleft['frame']==fnum]
k2_frame = c2_tracks_botleft[c2_tracks_botleft['frame']==fnum]
k3_frame = c3_tracks_botleft[c3_tracks_botleft['frame']==fnum]

fig, ax= plt.subplots()
a0 = plt.subplot2grid( (4,4), [0,1], 2, 2 )
a0.set_xlabel('K1')
a1 = plt.subplot2grid( (4,4), [2,0], 2, 2 )
a1.set_xlabel('K2')
a2 = plt.subplot2grid( (4,4), [2,2], 2, 2 )
a2.set_xlabel('K3')


def label_all_points(df, ax):
    xy = df.loc[:,['x','y']].to_numpy(dtype=np.float64)
    x, y = xy[:,0], xy[:,1]
    for xx,yy, pointlabel in zip(x,y, df['id']):
        ax.text(xx, yy, pointlabel, color='g')
        ax.scatter(xx, yy, facecolor='none', edgecolors='blue')
    ax.figure.canvas.blit(ax.bbox)

def plot_new_image(fnum):
    image_k1 = plt.imread(k1_images[fnum])
    image_k2 = plt.imread(k2_images[fnum])
    image_k3 = plt.imread(k3_images[fnum])
    
    a0.imshow(np.flipud(image_k1), aspect='equal', origin='lower')
    a0.figure.canvas.draw()
    a1.imshow(np.flipud(image_k2), aspect='equal', origin='lower')
    a1.figure.canvas.draw()
    a2.imshow(np.flipud(image_k3), aspect='equal', origin='lower')
    a2.figure.canvas.draw()
    
    fig.canvas.blit(a0.bbox)
    fig.canvas.blit(a1.bbox)
    fig.canvas.blit(a2.bbox)

def label_detections(fnum):
    k1_frame = c1_tracks_botleft[c1_tracks_botleft['frame']==fnum]
    label_all_points(k1_frame, a0)
    k2_frame = c2_tracks_botleft[c2_tracks_botleft['frame']==fnum]
    label_all_points(k2_frame, a1)
    k3_frame = c3_tracks_botleft[c3_tracks_botleft['frame']==fnum]
    label_all_points(k3_frame, a2)
    fig.canvas.blit(a0.bbox)
    fig.canvas.blit(a1.bbox)
    fig.canvas.blit(a2.bbox)


class fakeevent():
    pass

def framenum_entry(entryval):
    '''
    '''
    global fnum
    fnum = int(entryval)
    fakeevent.key = 'z'
    on_press(fakeevent)
    
    

def on_press(event):
    '''
    Keys to press are 'n' for next, 
    'b' for 'back' and 'h' for home or 0th frame.
    '''
    global fnum 
    #print('press', event.key)
    if event.key == 'n':
        fnum += 1 
    elif event.key == 'b':
        if fnum >=1:
            fnum -= 1 
        else:
            pass
    elif event.key == 'h':
        fnum =  0 

    a0.cla()
    a1.cla()
    a2.cla()
    
    try:
        plot_new_image(fnum)
        label_detections(fnum)
        a0.set_title(f'frame number: {fnum}')
        a0.set_xlabel('K1')
        a1.set_xlabel('K2')
        a2.set_xlabel('K3')

        a0.figure.canvas.draw()
        a1.figure.canvas.draw()
        a2.figure.canvas.draw()
    except:
        a0.set_title(f'Frame number: {fnum} invalid')
        pass
        

def on_click2(event):
    event_axes = [ax.in_axes(event) for ax in [a0, a1, a2]]
    if sum(event_axes) == 0:
        return None

    for each in [a0.lines, a1.lines, a2.lines]:
        for every in each:
            every.remove()

    source_camera = int(np.argwhere(event_axes)) + 1 
    print(f'Source camera is : {source_camera}')
    if source_camera == 1:
        C1 = c1_dlt.copy()
        C2 = c2_dlt.copy()
        C3 = c3_dlt.copy()
        ax_source = a0 
        ax_C2 = a1
        ax_C3 = a2

    elif source_camera == 2:
        C1 = c2_dlt.copy()
        C2 = c3_dlt.copy()
        C3 = c1_dlt.copy()
        ax_source = a1
        ax_C2 = a2
        ax_C3 = a0

    elif source_camera == 3:
        C1 = c3_dlt.copy()
        C2 = c1_dlt.copy()
        C3 = c2_dlt.copy()
        ax_source = a2
        ax_C2 = a0
        ax_C3 = a1
    
    u,v = event.xdata, event.ydata
    ax_source.plot(u,v,'w*')
    ax_source.figure.canvas.draw()
    ax_C2.figure.canvas.draw()
    ax_C3.figure.canvas.draw()
    
    
    
    m12 ,b12 = partialdlt(u, v, C1, C2)
    m13 ,b13 = partialdlt(u, v, C1, C3)
    x_lims = np.linspace(0, px*2, 100)
    epi_line_y12  = m12*x_lims + b12
    epi_line_y13  = m13*x_lims + b13
    
    undist_epiline_xy12 = cv2.undistortPoints(np.column_stack((x_lims, epi_line_y12)),
                                             Kteax, dist_coefs, P=Kteax).reshape(-1,2)
    undist_epiline_xy13 = cv2.undistortPoints(np.column_stack((x_lims, epi_line_y13)),
                                             Kteax, dist_coefs, P=Kteax).reshape(-1,2)
    

    valid_y12 = np.logical_and(undist_epiline_xy12[:,1]>0, undist_epiline_xy12[:,1] <2*py)
    valid_x12 = np.logical_and(undist_epiline_xy12[:,0]>0, undist_epiline_xy12[:,0] <2*px)
    valid_xy12 = np.logical_and(valid_y12, valid_x12)
    
    
    ax_C2.plot(undist_epiline_xy12[valid_xy12,0], undist_epiline_xy12[valid_xy12,1],
               'r', linewidth=0.5)
    
    valid_y13 = np.logical_and(undist_epiline_xy13[:,1]>0, undist_epiline_xy13[:,1] <2*py)
    valid_x13 = np.logical_and(undist_epiline_xy13[:,0]>0, undist_epiline_xy13[:,0] <2*px)
    valid_xy13 = np.logical_and(valid_x13, valid_y13)
    ax_C3.plot(undist_epiline_xy13[valid_xy13,0], undist_epiline_xy13[valid_xy13,1],
               'r', linewidth=0.5)
    ax_source.figure.canvas.draw()
    ax_C2.figure.canvas.draw()
    ax_C3.figure.canvas.draw()

axbox = fig.add_axes([0.2, 0.015, 0.4, 0.075])
framenum_box = TextBox(axbox, 'Go to Framenum')
framenum_box.on_submit(framenum_entry)

fig.canvas.mpl_connect('button_press_event', on_click2)
fig.canvas.mpl_connect('key_press_event', on_press)
#%%
from track2trajectory import camera

cm_mtrxs = []
cam_centres = []
cameras = []
for i,dltcoefs in enumerate([c1_dlt, c2_dlt, c3_dlt]):
    cmmtrx, Z, ypr = transformation_matrix_from_dlt(dltcoefs)
    camera_matrix = cmmtrx.T[:3,:]
    camcentre = cam_centre_from_dlt(dltcoefs)
    cam_centres.append(camcentre)
    cm_mtrxs.append(camera_matrix)
    cameras.append(camera.Camera(i+1, camcentre, fx, px, py, fx, fy, 
                                 Kteax, camera_matrix[:3,-1], camera_matrix[:3,:3],
                                 dist_coefs, camcentre, camera_matrix))

#%%
matchids = find_best_matches(dlt_coefs, c1_tracks_botleft, c2_tracks_botleft, c3_tracks_botleft)

match_ids, counts = np.unique(np.array(matchids), return_counts=True)
sort_inds = np.argsort(counts)[::-1]
matchids_sorted = match_ids[sort_inds]
sortcounts = counts[sort_inds]
valid_ids_counts = []
for matchid, count in zip(matchids_sorted, sortcounts):
    parts = matchid.split('-')
    numnans = parts.count('nan')
    if numnans < 2:
        valid_ids_counts.append((matchid, count))

frequent_idmatches = list(filter(lambda X: X[-1]>10, valid_ids_counts))

#%%
c1_tracks = c1_tracks_botleft.copy()
c2_tracks = c2_tracks_botleft.copy()
c3_tracks = c3_tracks_botleft.copy()

for (matchid, counts) in frequent_idmatches:
    cam1id, cam2id, cam3id = matchid.split('-')
    c1_tracks['id'] = c1_tracks['id'].replace(cam1id, matchid)
    c2_tracks['id'] = c2_tracks['id'].replace(cam2id, matchid)
    c3_tracks['id'] = c3_tracks['id'].replace(cam3id, matchid)

#%% Check for any potential conflicts in point assignments
all_match_sets = []
for (matchid, count) in valid_ids_counts:
    all_match_sets.append(set(matchid.split('-')))

intersecting = []
for i, target_set in enumerate(all_match_sets[:-1]):
    for test_set in all_match_sets[i+1:]:
        overlap = target_set.intersection(test_set)
        if len(overlap) > 0:
            intersecting.append((target_set, test_set))
#%%
c123_tracks = pd.concat([c1_tracks, c2_tracks, c3_tracks]).reset_index(drop=True)
frames = c123_tracks['frame'].unique()
obj_ids = c123_tracks['id'].unique()
trajectory_data = []
for fnum in frames:
    thisframe = c123_tracks[c123_tracks['frame']==fnum]
    for pointid, subdf in thisframe.groupby('id'):
        subdf = subdf.sort_values('cid')
        uv_coods = subdf.loc[:,['x','y']].to_numpy().flatten()
        caminds = subdf.loc[:,'cid'].to_numpy()-1
        x,y,z = row_wise_dlt_reconstruct(uv_coods, dlt_coefs[:,caminds])
        trajectory_data.append([fnum, pointid, x, y,z])
trajectories = pd.DataFrame(trajectory_data, columns=['frame', 'id', 'x','y','z'])
trajectories.to_csv('xyz_2018-08-17_first51frames.csv')

# #%%
# plt.figure()
# a0 = plt.subplot(111, projection='3d')
# for pointid, traj in trajectories.groupby('id'):
#     plt.plot(traj['x'], traj['y'], traj['z'],'*')
#%%
# And now check for any missing trajectories by overlaying the original tracks 
# by comparing the trajectories

def dlt_inv_wrapper(xyz, c):
    return dlt_inverse(c, xyz.reshape(-1,3))

traj_xyz = trajectories.loc[:,['x','y','z']].to_numpy()
pointids = trajectories.loc[:,'id'].to_numpy()
camera_uvs = {}
for camnum in range(3):
    uv_cam = np.apply_along_axis(dlt_inv_wrapper, 1, traj_xyz, dlt_coefs[:,camnum])
    camera_uvs[camnum] = uv_cam.reshape(-1,2)
#%%
import matplotlib.pyplot as plt

cmap = plt.get_cmap('viridis')
unique_pointids = np.unique(pointids)
colors = np.linspace(0,1,len(np.unique(pointids)))
pointid_to_color = {ptid: color for ptid, color in zip(unique_pointids, colors)}

pointcolors = [pointid_to_color[each] for each in pointids]
#%%
# Verify that the 3D point projections mirror the raw data across the cameras.

fig, ax= plt.subplots()
a01 = plt.subplot2grid( (4,4), [0,1], 2, 2 )
a01.set_xlabel('K1')
a02 = plt.subplot2grid( (4,4), [2,0], 2, 2 )
a02.set_xlabel('K2')
a03 = plt.subplot2grid( (4,4), [2,2], 2, 2 )
a03.set_xlabel('K3')

a01.scatter(c1_tracks_botleft.loc[:,'x'], c1_tracks_botleft.loc[:,'y'], edgecolors='r',
            marker='o', s=40, facecolors='none', )
a01.scatter(camera_uvs[0][:,0], camera_uvs[0][:,1], c=pointcolors, marker='+')
a01.set_ylim(0,512); a01.set_xlim(0,640)

a02.scatter(c2_tracks_botleft.loc[:,'x'], c2_tracks_botleft.loc[:,'y'], edgecolors='r',
            marker='o', s=40, facecolors='none', )
a02.scatter(camera_uvs[1][:,0], camera_uvs[1][:,1], c=pointcolors, marker='+')
a02.set_ylim(0,512);a02.set_xlim(0,640)

a03.scatter(c3_tracks_botleft.loc[:,'x'], c3_tracks_botleft.loc[:,'y'], edgecolors='r',
            marker='o', s=40, facecolors='none', )
a03.scatter(camera_uvs[2][:,0], camera_uvs[2][:,1], c=pointcolors, marker='+')
a03.set_ylim(0,512);a03.set_xlim(0,640)

#%%
# Now convert the xyz points from camera coordinate system to LiDAR coordinate system. 

# This is the transformation matrix for 2018-08-17 from Julian's thesis. 
A = np.array(([-0.7533, 0.6353, -0.1699, -1.7994],
              [-0.6575, -0.7332, 0.1734, 1.7945],
              [-0.0144, 0.2424, 0.9701, 0.2003]))
# Bring the 3d points in camera frame to LiDAR frame. 

traj_xyz_homog = np.column_stack((trajectories.loc[:,['x','y','z']].to_numpy(), np.ones(trajectories.shape[0])))
traj_lidarframe = np.apply_along_axis(lambda X: np.matmul(A, X), 1, traj_xyz_homog )[:,:-1]
trajectories_lidarframe = trajectories.copy()
trajectories_lidarframe.loc[:,['x','y','z']] = traj_lidarframe


