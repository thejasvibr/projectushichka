# -*- coding: utf-8 -*-
"""
Video-assisted acoustic tracking 
================================
Here I explore the idea of using the flight trajectories from the video tracking
to generate TDOA predictions - and 'search' for these in the audio. I'll first work with
the speaker playbacks of 2018-08-17 because this is a controlled way of checking how
well the method works. If the video tracking is bad - it will automatically lead 
to poor TDOA estimates. 

The chirp playback audio
~~~~~~~~~~~~~~~~~~~~~~~~
The moving speaker plays back what I call the '9-chirp' audio file - with 
downward linear and  logarithmic, along with bidirectional sweeps of 6, 12 and
24 ms duration - with the same start and end frequencies. Each sound-type is
designed such that it is at the end of a 200 ms audio window. i.e. for the 5 ms
durations, there is a silence of 195 ms, and for the 20 ms sounds - there is a 
silence of 180 ms. 


"""
import glob
import matplotlib.pyplot as plt 
plt.rcParams['agg.path.chunksize'] = 10000
import numpy as np 
import os 
import pandas as pd
import scipy.interpolate as interpolate
import soundfile as sf
import sys 
sys.path.append('../../ushichka_tools/')
import scipy.signal as signal 
import audio_handling
from localisation_code import localisation_mpr2003 as lmpr
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
time_interp_xyz['t'] = new_t

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
mic_xyz['id'] = mic_ids
#%% The video 3D triangulation is not bad at all. The central-peripheral mic-to-mic
# distances should be 1.2 m, and here we see 1.14 - 1.17 m. This corresponds to about
# just 5-8 % error. 
sankens = mic_xyz.loc[mic_xyz.index[-4:],'x':'z'].to_numpy()
dist_mat = spatial.distance_matrix(sankens, sankens)
print(dist_mat[:,0])

#%%
# Now let's calculate the speaker-to-mic distances over time using the cubic interpolated
# trajectory. 
vsound = 343.0 # m/s
mic_speaker_distances ={i: (spatial.distance_matrix(sankens[i,:].reshape(-1,3), time_interp_xyz.loc[:,:'z'])).flatten() for i in range(4)}
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
sync_audio = audio[first_frame:,7]
plt.figure()
plt.specgram(sanken_audio[:,0], Fs=fs)

#%% 
# Using time-of-flight to access audio in the relevant recording portion. 
# We need to calculate emission time starting from 0.18 seconds because
emission_time = 0.180
tof_pbk0 = tracking_data.loc[(tracking_data.t_interp-emission_time).abs().idxmin(),:]

#%%
# When does the first playback arrive at mic0?

b,a = signal.butter(4, 40000/fs, 'highpass')
first_pbk = sanken_audio[slice(int((emission_time-0.005)*fs), int((emission_time+0.025)*fs)),:]
first_pbk = np.apply_along_axis(lambda X: signal.filtfilt(b,a,X), 0, first_pbk)

crosscor = [signal.correlate(first_pbk[:,ch1], first_pbk[:,ch0]) for (ch1,ch0) in [(1,0), (2,0), (3,0)]]
all_pks = [signal.find_peaks(crosscor[i], height=5e-4, distance=int(fs*0.0001)) for i in range(3)]
all_tde = []
for each in all_pks:
    max_val_index = each[1]['peak_heights'].argmax()
    tde = (each[0][max_val_index] - first_pbk.shape[0])/fs
    all_tde.append(tde)

#%%
# 4-channel localisation gives 2 solutions - choose the one that makes more sense. 
sound_source = lmpr.mellen_pachter_raquet_2003(sankens, np.array(all_tde)*vsound)
prob_source = sound_source[0]
# what is the array-source distance based on acoustic localisation
source_array_dists = spatial.distance_matrix(prob_source.reshape(-1,3), sankens)


#%%
mic_toas = [emission_time+tof_pbk0[f'tof{each}'] for each in range(4)]
mic_toas_ind = [slice(int(i*fs), int(i*fs)+int(0.006*fs)) for i in mic_toas]

plt.figure()
a0 = plt.subplot(211)
plt.specgram(sanken_audio[:int(0.25*fs),0], Fs=fs)
plt.vlines([mic_toas_ind[0].start/fs, mic_toas_ind[0].stop/fs], 10000, 80000, 'r')
plt.vlines([mic_toas_ind[0].start/fs, mic_toas_ind[0].stop/fs], 0, 0.01, 'r')
plt.subplot(212, sharex=a0)
plt.plot(np.linspace(0,0.25,int(fs*0.25)),sync_audio[:int(fs*0.25)])

#%%
# Performing REAL multichannel audio-video checking
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's also check the acoustic localisation when we include some other microphones
# into the audio. I believe the order of mic_ids to channels was:
# channels 0-6 : SMP1,7, channel 7: sync signal, channel 8-11: SANKEN 9-12.
# channel 13-16 : empty., channel 17-23: SMP 8, SMP1p-SMP8p. In this case, I think 
# (Based on notes from Field Equipment log notebook - entry on 9/11/2018)
#
# Remember the 28 channels from the speaker playback file are effectively 
# all the available channels from the Fireface 802 (16 channels: 12 analog + 2 SPDIF + 2 ADAT})
# and Fireface UC (12 channels: 8 analog + 2 SPDIF + 2 ADAT)
#
# Abbrieviations
#  Ff802 - Fireface 802
#  FfUC - Fireface UC
#  Focusrite - Focusrite Scarlett OctoPre
# 
# In effect the microphone id order guess is:
# | Channel #     |  Channel TYPE   | Audio source    |
# |---------------|-----------------|-----------------| 
# |    0          | Audio/XLR       |      1/SMP1     |
# |    1          | Audio/XLR       |      2/SMP2     |
# |    2          | Audio/XLR       |      3/SMP3     |
# |    3          | Audio/XLR       |      4/SMP4     |
# |    4          | Audio/XLR       |      5/SMP5     |
# |    5          | Audio/XLR       |      6/SMP6     |
# |    6          | Audio/XLR       |      7/SMP7     |
# |    7          | Audio/XLR       |sync Ff802       |
# |    8          | Audio/XLR       |   9/SANKEN 9    |
# |    9          | Audio/XLR       |   10/SANKEN 10  |
# |   10          | Audio/XLR       |   11/SANKEN 11  |
# |   11          | Audio/XLR       |   12/SANKEN 12  |
# |   12          | SPDIF           |   ---------     |
# |   13          | SPDIF           |   ---------     |
# |   14          | ADAT            |   8/SMP8        |
# |   15          | ADAT            |   1p/SMP1p      |
# |   16          | Audio/XLR       |   2p/SMP2p      |
# |   17          | Audio/XLR       |   3p/SMP3p      |
# |   18          | Audio/XLR       |   4p/SMP4p      |
# |   19          | Audio/XLR       |   5p/SMP5p      |
# |   20          | Audio/XLR       |   6p/SMP6p      |
# |   21          | Audio/XLR       |   7/SMP7p       |
# |   22          | Audio/XLR       |   8p/SMP8p      |
# |   23          | Audio/XLR       |sync  FfUC       |
# |   24          | SPDIF           |   ---------     |
# |   25          | SPDIF           |   ---------     |
# |   26          | ADAT            |sync Focusrite   |
# |   27          | ADAT            |playback/ empty  |
#
# ADAT channels didn't work on 2018-08-17
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# There is no audio data on the ADAT channels across both devices - field notes
# suggest may be because of the moisture.
#
#
# 10 channel attempt - only Fireface 802
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's keep things simple and only use 10 channels across one audio interface
# This reduces the effort required to sync the two audio interfaces.
# Channel 6 (0 index) has poor audio quality, so let's not use it. 

mic_inds = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]
tench_audio = audio[first_frame:, mic_inds]
# skip microphone 7 
mic_ids_to_choose = [str(i) for i in range(1,7)] + [str(j) for j in range(9,13)]
tench_mic_xyz = mic_xyz[mic_xyz['id'].isin(mic_ids_to_choose)].reset_index(drop=True)
tench_xyz = tench_mic_xyz.loc[:,'x':'z'].to_numpy()

#%%
# Checking field mic-to-mic and camera triangulation measurements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here we can see the field measurements made with a laser range-finder
# and the camera triangulation. The mic-to-mic distances are *pretty* spot-on
# with a max of ~12 cm difference! The camera based distances are systematically
# less than the field measurements. 
# The field mic-to-mic distances are from 2018-08-17 notebook entries, the average of
# 2 measurements 
field_measurements = np.array([3.635, 3.553, 3.554, 3.177,
                      3.136, 3.091, 0, 1.2, 1.2, 1.2])
mic2mic_tench = spatial.distance_matrix(tench_xyz, tench_xyz) # 
sanken9_distances = np.around(mic2mic_tench[:,6], 3)

field_n_cam_difference = sanken9_distances - field_measurements 
print(field_n_cam_difference)
relative_error = field_n_cam_difference/field_measurements

#%% 
# Now calculate the speaker-mic distances across time, and obtain the expected
# time-of-flights. Set the SANKEN 9 as the origin (microphone 0) - because this 
# is a useful reference point, and allows comparison for the only-sankens acoustic
# tracking. 

tench_snkn9_origin = np.row_stack((tench_xyz[6,:], tench_xyz[[0,1,2,3,4,5,7,8],:]))
tench_speaker_distances ={i: (spatial.distance_matrix(tench_snkn9_origin[i,:].reshape(-1,3), time_interp_xyz.loc[:,:'z'])).flatten() for i in range(tench_snkn9_origin.shape[0])}
tench_rangdiff_m0 = [tench_speaker_distances[i]-tench_speaker_distances[0] for i in range(1,tench_snkn9_origin.shape[0])] # mic2mic distances, re mic0
tench_snkn9_tde = pd.DataFrame(data=np.array(tench_rangdiff_m0).T/vsound)
tench_snkn9_tde['t'] = new_t
tench_snkn9_tof =  pd.DataFrame(data=tench_speaker_distances)/vsound
tench_snkn9_tof['t'] = new_t

# Also alter audio data to sanken9 as first channel. 
tench_snkn9_audio = np.column_stack((tench_audio[:,6], tench_audio[:,[0,1,2,3,4,5,7,8,9]]))














