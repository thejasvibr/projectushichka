# -*- coding: utf-8 -*-
"""
Manual trajectory matching attempts
===================================
Since my initial attempts at automated trajectory matching with track2trajectory 
failed, I'm now being forced to try it out manually. This module creates a
workflow for manual trajectory matching. 

Attention
~~~~~~~~~
The 2D tracking 'x' and 'y' are actually the row and column - these are therfore
switched in this module!


Environment to activate: tproject

Created on Mon Mar 20 13:20:38 2023
"""

# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from track2trajectory.projection import calcFundamentalMatrix



#%% Load the camera 2D points
all_cam_2d = pd.read_csv('all_camera_tracks_2018-08-17_P01_7000_first25frames.csv').loc[:,'id':]
all_cam_2d.columns = ['oid', 'frame', 'x', 'y', 'cid']
c1_tracks = all_cam_2d[all_cam_2d['cid']=='K2'].reset_index(drop=True)
c2_tracks = all_cam_2d[all_cam_2d['cid']=='K3'].reset_index(drop=True)

# IMPORTANT - this is NOT actually x,y - but row, col
c1_2d = c1_tracks.loc[:,['x','y']].to_numpy() 
c2_2d = c2_tracks.loc[:,['x','y']].to_numpy()

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

# #%%
# # apply undistortion now OR NOT - it seems to complicate things for now. 
cam1_undist = cv2.undistortPoints(np.fliplr(c1_2d), Kteax, dist_coefs, P=Kteax).reshape(-1,2)
cam2_undist = cv2.undistortPoints(c2_2d, Kteax, dist_coefs, P=Kteax).reshape(-1,2)

# c1_tracks_botleft['x'] = cam1_undist[:,1]
# c1_tracks_botleft['y'] = cam1_undist[:,0]
# c2_tracks_botleft['x'] = cam2_undist[:,1]
# c2_tracks_botleft['y'] = cam2_undist[:,0]
# # 

# plt.figure()
# plt.subplot(121)
# plt.plot(c1_2d[:,1], c1_2d[:,0],'*')
# plt.subplot(122)
# plt.plot(c2_2d[:,1], c2_2d[:,0],'*')
# # plt.plot(cam1_undist[:,1], cam1_undist[:,0],'r*')
# plt.grid()

#%% Now initialise camera objects with their projection matrices, along with 
# the F matrix that translates 2D points from one camera to the other. 
dlt_coefs = pd.read_csv('2018-08-17/2018-08-17_wand_dltCoefs_round4.csv', header=None).to_numpy()
c1_dlt, c2_dlt, c3_dlt  = [dlt_coefs[:,i] for i in range(3)]
def extract_P_from_dlt_v2(dltcoefs):
    '''No normalisation
    '''
    dltcoefs = np.append(dltcoefs, 1)
    dltcoefs = dltcoefs

    P = dltcoefs.reshape(3,4)
    return P

# generate projection matrix 
Pcam1 = extract_P_from_dlt_v2(c1_dlt)
Pcam2 = extract_P_from_dlt_v2(c2_dlt)


#%% Something's not working - why?
from track2trajectory.match2d import get_epiline
from track2trajectory.camera import Camera

# camera image is 640 x 512
px,py = 320, 256
fx, fy = 526, 526 # in pixels

transl_cam1 = []
rot_cam1 = []
Kteax = np.array([[fx, 0, px],
                  [0, fy, py],
                  [0, 0,  1]])

p1, p2 = np.float32([0,0]) # tangential distortion
k1, k2, k3 = np.float32([-0.3069, 0.1134, 0]) # radial distortion
dist_coefs = np.array([k1, k2, p1, p2, k3]) #in the opencv format
cam1 = Camera(0, [None, None] ,fx, px, py, fx, fy, Kteax, transl_cam1, rot_cam1, dist_coefs,
              [None]*3, Pcam1)

cam2 = Camera(0, [None, None] ,fx, px, py, fx, fy, Kteax, transl_cam1, rot_cam1, dist_coefs,
              [None]*3, Pcam2)
F = calcFundamentalMatrix(cam1, cam2)


#%%
from track2trajectory.dlt_to_world import partialdlt
global fnum
fnum = 0
k1_frame = c1_tracks_botleft[c1_tracks_botleft['frame']==fnum]
k2_frame = c2_tracks_botleft[c2_tracks_botleft['frame']==fnum]

fig, ax= plt.subplots()
a0 = plt.subplot(121)
a1 = plt.subplot(122, sharex=a0, sharey=a0)
fig.canvas.blit(a0.bbox)
fig.canvas.blit(a1.bbox)

def label_all_points(df, ax):
    xy = df.loc[:,['x','y']].to_numpy(dtype=np.float64)
    x, y = xy[:,0], xy[:,1]
    for xx,yy, pointlabel in zip(x,y, df['oid']):
        ax.text(xx, yy, pointlabel)
        ax.scatter(xx, yy, facecolor='none', edgecolors='blue')
    ax.figure.canvas.blit(ax.bbox)

def plot_new_image(fnum):
    image_k1 = plt.imread(f'cleaned_imgs/K2/cleaned_{fnum}.png')
    image_k2 = plt.imread(f'cleaned_imgs/K3/cleaned_{fnum}.png')
    a0.imshow(np.flipud(image_k1), aspect='equal', origin='lower')
    a0.figure.canvas.draw()
    a1.imshow(np.flipud(image_k2), aspect='equal', origin='lower')

def label_detections(fnum):
    k1_frame = c1_tracks_botleft[c1_tracks_botleft['frame']==fnum]
    label_all_points(k1_frame, a0)
    k2_frame = c2_tracks_botleft[c2_tracks_botleft['frame']==fnum]
    label_all_points(k2_frame, a1)

def on_press(event):
    global fnum 
    #print('press', event.key)
    if event.key == 'n':
        fnum += 1 
        a1.set_title('')
    elif event.key == 'b':
        fnum -= 1 
        a1.set_title('')
    else:
        a1.set_title(f'{event.key} is unrecognised - only n and b')
    a0.cla()
    a1.cla()
    try:
        plot_new_image(fnum)
        label_detections(fnum)
        a0.set_title(f'frame number: {fnum}')
        a1.figure.canvas.draw()
        a0.figure.canvas.draw()
    except:
        a0.set_title(f'{fnum} is invalid - unable to plot')
        a0.cla()
        a0.cla()

def on_click2(event):
    u,v = event.xdata, event.ydata
    #    print(x,y)
    m ,b = partialdlt(u, v, c2_dlt, c3_dlt)
    x_lims = np.linspace(0, px*2, 10)
    epi_line_y  = m*x_lims + b
    for each in a1.lines:
        each.remove()
    valid_y = np.logical_and(epi_line_y>=0, epi_line_y<=2*py)
    a1.plot(x_lims[valid_y], epi_line_y[valid_y], 'r')
    a1.figure.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_click2)
fig.canvas.mpl_connect('key_press_event', on_press)


plot_new_image(0)
a0.set_ylim(0,py*2);plt.xlim(0,px*2)
label_detections(0)
# label_all_points(k1_frame, a0)
# label_all_points(k2_frame, a1)


#%%








