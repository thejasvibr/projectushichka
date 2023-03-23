# -*- coding: utf-8 -*-
"""
Video-assisted acoustic tracking 
================================
Here I explore the idea of using the flight trajectories from the video tracking
to generate TDOA predictions - and 'search' for these in the audio. I'll first work with
the speaker playbacks of 2018-08-17 because this is a controlled way of checking how
well the method works. If the video tracking is bad - it will automatically lead 
to poor TDOA estimates. 

"""
import glob
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import scipy

#%% Load the 3D video trajectories 
vid_3d_file = glob.glob('../2018-08-17/speaker_playbacks/*xyzpts*.csv')[0]
vid_3d = pd.read_csv(vid_3d_file)
vid_3d.columns = ['x','y','z']
# remove all frames from 700th onwards
vid_3d = vid_3d.loc[:700,:].dropna()
vid_3d['frame'] = vid_3d.index

plt.figure()
a0 = plt.subplot(111, projection='3d')
plt.plot(vid_3d['x'], vid_3d['y'], vid_3d['z'], '*')

#%% interpolate the ~0.5 s interval digitised xyz coordinates to 25 Hz. 





