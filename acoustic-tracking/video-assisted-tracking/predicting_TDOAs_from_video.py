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
import os 
import pandas as pd
import scipy.interpolate as interpolate
import soundfile as sf
import sys 
sys.path.append('../../ushichka_tools/')
import audio_handling
#%% Load the 3D video trajectories 
vid_3d_file = glob.glob('../2018-08-17/speaker_playbacks/*xyzpts*.csv')[0]
vid_3d = pd.read_csv(vid_3d_file)
vid_3d.columns = ['x','y','z']
# remove all frames from 700th onwards
vid_3d = vid_3d.loc[:700,:].dropna()
vid_3d['frame'] = vid_3d.index
vid_3d['t'] = vid_3d.index*0.04

plt.figure()
a0 = plt.subplot(111, projection='3d')
plt.plot(vid_3d['x'], vid_3d['y'], vid_3d['z'], '*')

#%% interpolate the ~0.5 s interval digitised xyz coordinates to 25 Hz. 
cubic_interp = {axis: interpolate.CubicSpline(vid_3d.loc[:,'t'], vid_3d.loc[:, axis], ) for axis in ['x','y','z']}
new_t = np.arange(0, max(vid_3d['t'])+0.005, 0.005)
time_interp_xyz = pd.DataFrame(data= {axis: cubic_interp[axis](new_t) for axis in ['x', 'y', 'z']})

plt.figure()
a1 = plt.subplot(111, projection='3d')
plt.plot(time_interp_xyz['x'], time_interp_xyz['y'], time_interp_xyz['z'])

#%% Now load the microphone points - generated with round4 DLT coefficients
# 
# mic_xy_path = 'D:/thermal_camera/2018-08-17/mic_and_wall_points/2018-08-17/mic_2D_and_3D/'
# mic_xy_short = pd.read_csv(os.path.join(mic_xy_path,'DLTdv7_data_2018-08-17_P02_micpointsxypts.csv')).dropna()
# mic_xy_short.to_csv('DLTdv7_2018-08-17_custom_mic2Dxypts.csv', index=False)

mic_xyz = pd.read_csv('DLTdv8_data_2018-08-17_mics_3camxyzpts.csv').dropna()
mic_xyz.columns = ['x', 'y', 'z']

plt.figure()
a1 = plt.subplot(111, projection='3d')
plt.plot(mic_xyz['x'], mic_xyz['y'], mic_xyz['z'], 'r*')
plt.plot(vid_3d['x'], vid_3d['y'], vid_3d['z'], '*')

#%% 1-8 are the older SMP mics, the 1-p, 2-p are the 1', 2' etc, which are 
# the newer SMP microphones placed in the next round. Mics 9-12 are the Sanken mics
import scipy.spatial as spatial

mic_ids = ['1', '2','3', '4', '5', '6', '7', '8', '1-p', '2-p', '3-p',
           '4-p', '5-p', '6-p', '7-p', '8-p', '9', '10', '11', '12' ]

#%% The video 3D triangulation is not bad at all. The central-peripheral mic-to-mic
# distances should be 1.2 m, and here we see 1.14 - 1.17 m. This corresponds to about
# just 5-8 % error. 
sankens = mic_xyz.loc[mic_xyz.index[-4:],:].to_numpy()
dist_mat = spatial.distance_matrix(sankens, sankens)
print(dist_mat[:,0])

#%%
# Now let's calculate the speaker-to-mic distances over time using the cubic interpolated
# trajectory. 
vsound = 343.0 # m/s
mic_speaker_distances ={i: (spatial.distance_matrix(sankens[i,:].reshape(-1,3), time_interp_xyz.loc[:,:])).flatten() for i in range(4)}
m123_0 = [mic_speaker_distances[i]-mic_speaker_distances[0] for i in range(1,4)] # mic2mic distances, re mic0
tde_123_0 = np.array(m123_0).T/vsound
time_of_flight =  pd.DataFrame(data=mic_speaker_distances)/vsound

time_of_flight.columns = ['tof_0','tof_1','tof_2','tof_3']
tracking_data = pd.concat((pd.DataFrame(new_t), pd.DataFrame(tde_123_0), time_of_flight), axis=1)
tracking_data.columns = ['t_interp', 'tde10', 'tde20' , 'tde30', 'tof0', 'tof1', 'tof2', 'tof3']
#%%
# Load the speaker playback audio file - just the Sanken mics - to keep it simple for now, and avoid time-sync steps etc. 
audio_folder = 'E:/fieldword_2018_002/actrackdata/wav/2018-08-17_003/'
playback_file = 'SPKRPLAYBACK_multichirp_2018-08-18_09-15-06.WAV'
fs = 192000
audio, fs = sf.read(os.path.join(audio_folder, playback_file), stop=6*fs)
#%%
# get the first video frame, and keep only audio from there on. 
first_frame = audio_handling.first_frame_index(audio[:,7])

plt.figure()
plt.plot(audio[:384000,7])
plt.plot(first_frame, audio[first_frame,7], 'r*')
#%%
# The 9 chirps are emitted at the source with 200 ms gap. The sound ends at the 200 ms
# window. 
sanken_audio = audio[first_frame:,8:12] # load the SANKEN mics on the tristar array frame. 
plt.figure()
plt.specgram(sanken_audio[:,0], Fs=fs)



