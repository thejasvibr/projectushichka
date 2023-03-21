# -*- coding: utf-8 -*-
"""
Manual trajectory matching attempts
===================================
Since my initial attempts at automated trajectory matching with track2trajectory 
failed, I'm now being forced to try it out manually. This module creates a
workflow for manual trajectory matching. 

A mouse click on any one camera image generates the epipolar lines across the two
other cameras.




Attention
~~~~~~~~~
The 2D tracking 'x' and 'y' are actually the row and column - these are therfore
switched in this module!

Previously run module
~~~~~~~~~~~~~~~~~~~~~
traj_correction.py

Environment to activate: tproject

Created on Mon Mar 20 13:20:38 2023
"""

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%% Load the camera 2D points
all_cam_2d = pd.read_csv('all_camera_tracks_2018-08-17_P01_7000_first25frames.csv').loc[:,'id':]
all_cam_2d.columns = ['oid', 'frame', 'x', 'y', 'cid']
c1_tracks = all_cam_2d[all_cam_2d['cid']=='K1'].reset_index(drop=True)
c2_tracks = all_cam_2d[all_cam_2d['cid']=='K2'].reset_index(drop=True)
c3_tracks = all_cam_2d[all_cam_2d['cid']=='K3'].reset_index(drop=True)

# IMPORTANT - this is NOT actually x,y - but row, col
c1_2d = c1_tracks.loc[:,['x','y']].to_numpy() 
c2_2d = c2_tracks.loc[:,['x','y']].to_numpy()
c3_2d = c3_tracks.loc[:,['x','y']].to_numpy() 

#Load the dlt coefficients - and infer the Projection matrix. We already
# know the intrinsic matrix (common to all cameras)

# camera image is 640 x 512
px,py = 320, 256
fx, fy = 526, 526 # in pixels

Kteax = np.array([[fx, 0, px],
                  [0, fy, py],
                  [0, 0,  1]])

p1, p2 = np.float32([0,0]) # tangential distortion
k1, k2, k3 = np.float32([-0.3069, 0.1134, 0]) # radial distortion
dist_coefs = np.array([k1, k2, p1, p2, k3]) #in the opencv format
# Shift the origin from top left to lower left OTHERWISE THE DLT COEFS DONT MAKE SENSE
c1_2d[:,0] = py*2 - c1_2d[:,0] 
c2_2d[:,0] = py*2 - c2_2d[:,0]
c3_2d[:,0] = py*2 - c3_2d[:,0]

c1_tracks_botleft = c1_tracks.copy()
c1_tracks_botleft['x'] = c1_2d[:,1]
c1_tracks_botleft['y'] = c1_2d[:,0]
c1_tracks_botleft['oid'] = 'k1_'+c1_tracks_botleft['oid'].astype(str) 
c1_tracks_botleft['cid'] = 1

c2_tracks_botleft = c2_tracks.copy()
c2_tracks_botleft['x'] = c2_2d[:,1]
c2_tracks_botleft['y'] = c2_2d[:,0]
c2_tracks_botleft['oid'] = 'k2_'+c2_tracks_botleft['oid'].astype(str) 
c2_tracks_botleft['cid'] = 2


c3_tracks_botleft = c3_tracks.copy()
c3_tracks_botleft['x'] = c3_2d[:,1]
c3_tracks_botleft['y'] = c3_2d[:,0]
c3_tracks_botleft['oid'] = 'k3_'+c3_tracks_botleft['oid'].astype(str) 
c3_tracks_botleft['cid'] = 3

# # #%%
# # # apply undistortion now OR NOT - it seems to complicate things for now. 
# cam1_undist = cv2.undistortPoints(np.fliplr(c1_2d), Kteax, dist_coefs, P=Kteax).reshape(-1,2)
# cam2_undist = cv2.undistortPoints(c2_2d, Kteax, dist_coefs, P=Kteax).reshape(-1,2)


#%% Now initialise camera objects with their projection matrices, along with 
# the F matrix that translates 2D points from one camera to the other. 
dlt_coefs = pd.read_csv('2018-08-17/2018-08-17_wand_dltCoefs_round4.csv', header=None).to_numpy()
c1_dlt, c2_dlt, c3_dlt  = [dlt_coefs[:,i] for i in range(3)]

#%%
from track2trajectory.dlt_to_world import partialdlt
global fnum
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
    for xx,yy, pointlabel in zip(x,y, df['oid']):
        ax.text(xx, yy, pointlabel)
        ax.scatter(xx, yy, facecolor='none', edgecolors='blue')
    ax.figure.canvas.blit(ax.bbox)

def plot_new_image(fnum):
    image_k1 = plt.imread(f'cleaned_imgs/K1/cleaned_{fnum}.png')
    image_k2 = plt.imread(f'cleaned_imgs/K2/cleaned_{fnum}.png')
    image_k3 = plt.imread(f'cleaned_imgs/K3/cleaned_{fnum}.png')
    
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

def on_press(event):
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

    ax_source.figure.canvas.draw()
    ax_C2.figure.canvas.draw()
    ax_C3.figure.canvas.draw()
    
    
    u,v = event.xdata, event.ydata
    m12 ,b12 = partialdlt(u, v, C1, C2)
    m13 ,b13 = partialdlt(u, v, C1, C3)
    x_lims = np.linspace(0, px*2, 10)
    epi_line_y12  = m12*x_lims + b12
    epi_line_y13  = m13*x_lims + b13
    

    valid_y12 = np.logical_and(epi_line_y12>=0, epi_line_y12 <=2*py)
    ax_C2.plot(x_lims[valid_y12], epi_line_y12[valid_y12], 'r', linewidth=0.5)
    
    
    valid_y13 = np.logical_and(epi_line_y13>=0, epi_line_y13 <=2*py)
    ax_C3.plot(x_lims[valid_y13], epi_line_y13[valid_y13], 'r', linewidth=0.5)
    ax_source.figure.canvas.draw()
    ax_C2.figure.canvas.draw()
    ax_C3.figure.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_click2)
fig.canvas.mpl_connect('key_press_event', on_press)


plot_new_image(0)
a0.set_ylim(0,py*2);plt.xlim(0,px*2)
label_detections(0)
# label_all_points(k1_frame, a0)
# label_all_points(k2_frame, a1)


#%%
c1_ids = c1_tracks_botleft['oid'].unique()
cam_corresps =   pd.DataFrame(data={'c1_oid': ['k1_1.0', 'k1_2.0', 'k1_6.0',
                                               'k1_19.0', 'k1_26.0', 'k1_27.0',
                                               'k1_45.0', 'k1_34.0', np.nan,
                                               np.nan,    np.nan   , np.nan,
                                               np.nan,    np.nan],
                                    'c2_oid': ['k2_1.0', 'k2_2.0', 'k2_4.0',
                                               'k2_6.0',  'k2_8.0', 'k2_5.0',
                                               'k2_11.0', np.nan, 'k2_10.0',
                                               'k2_15.0', 'k2_16.0', 'k2_3.0',
                                               'k2_12.0', 'k2_17.0'],
                                    'c3_oid': ['k3_4.0', 'k3_5.0',  np.nan,
                                               'k3_41.0', 'k3_36.0', 'k3_21.0',
                                               'k3_43.0', np.nan, np.nan,
                                               'k3_44.0', np.nan , np.nan,
                                               np.nan   , np.nan]})
def fuse_point_ids(correspondences):
    '''
    Parameters
    ----------
    correspondences: pd.DataFrame
        With columns 'c1_oid, c2_oid, c3_oid'

    Returns 
    -------
    with_fusedid: pd.DataFrame
        Copy of correspondences with a new column holding the 
        fused-point IDs. The fused-IDs are just the c1,2,3 object
        ids separted by an underscore ('_')
    '''
    with_fusedid = correspondences.copy()
    with_fusedid['fused_id'] = with_fusedid['c1_oid'].astype(str) + '-'+ with_fusedid['c2_oid'].astype(str) + '-' + with_fusedid['c3_oid'].astype(str)
    return with_fusedid

# check to see if there are some other correspondences to be made. 
c2_notmatched = set(c2_tracks_botleft['oid'].unique()) - set(cam_corresps['c2_oid'])
c3_notmatched = set(c3_tracks_botleft['oid'].unique()) - set(cam_corresps['c3_oid'])

c2_tracks_botleft.groupby('oid').get_group('k2_17.0')
c3_tracks_botleft.groupby('oid').get_group('k3_4.0')

#%% 
# Assign new point ids to the old ids
matched_ids = fuse_point_ids(cam_corresps)

def assign_new_point_id(matched_ids, cam_df):
    '''
    Parameters
    ----------
    matched_ids : pd.DataFrame
        With at least there columns: c1_oid, c2_oid, c3_oid, fused_id
    cam_df : pd.DAtaFrame
        With at least these columns: oid, frame, x, y, cid
        Where oid is object id and cid is camera id
    
    Returns 
    -------
    pd.Series?
    '''
    df_copy = cam_df.copy()
    df_copy['point_id'] = df_copy['oid'].copy()
    # replace the oid with the fused id
    for i, row in df_copy.iterrows():
        matched_ids_rowcol = np.argwhere((matched_ids == row['oid']).to_numpy()).flatten()
        matched_ids_row = matched_ids_rowcol[0]
        df_copy.loc[i,'point_id'] = matched_ids.loc[matched_ids_row, 'fused_id']
    return df_copy


cam1_points = assign_new_point_id(matched_ids, c1_tracks_botleft)
cam2_points = assign_new_point_id(matched_ids, c2_tracks_botleft)
cam3_points = assign_new_point_id(matched_ids, c3_tracks_botleft)
all_cam_points = pd.concat([cam1_points, cam2_points, cam3_points]).reset_index(drop=True)
all_cam_points.to_csv('matched_2018-08-17_P01_7000_first25frames.csv')