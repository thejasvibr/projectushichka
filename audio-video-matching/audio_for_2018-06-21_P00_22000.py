"""
Trying to find the audio recording for 2018-06-21 
==================================================
On 2018-06-21  the first session P000 on cameras, and 2018-06-21_001
for audio data - had 26 audio files and 27 video files - which means one
of the video fiels is a false trigger probably done manually. 

Here, we'll see which video file is the odd one out. 


Summary
~~~~~~~
00019000.TMC is the odd one out, with 2.64 s duration - that doesn't correspond
to any of the audio file durations. All of the other audio and video file
durations match exactly. 

Exporting meta data for 2018-06-21
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TMC data from K2/P000 exported with the command

>>>.\\ThermoViewer.exe -folder F:\\fieldwork_2018_001\\video\\2018-06-21\\K2\\P0000000
 -expa C:\\Users\\theja\\Documents\\research_repos\\projectushichka\\audio-video-matching\\2018-06-21_video_meta
  -exmeta CSVfa -exfn 2018-06-21_ -exfo avi

"""
import glob
from natsort import natsorted
import matplotlib.pyplot as plt 
plt.rcParams['agg.path.chunksize'] = 100000
import numpy as np
import pandas as pd 
import soundfile as sf
import scipy.ndimage as ndi

#%% Use the previously generated meta-data only files to access # frames in all
# TMC files. 
all_csvfiles = natsorted(glob.glob('2018-06-21_video_meta/*.csv'))

video_durations = np.array([pd.read_csv(each, delimiter=';').shape[0] for each in all_csvfiles])/25
print(f'Video file durations are: {video_durations}')

print(f'Total video recording time {np.sum(video_durations)} s')

#%% Get an estimate of the total audio recording time in the first session. 

audio_filefolder = 'E:\\fieldwork_2018_001\\actrackdata\\wav\\2018-06-21_001\\'
audio_files = natsorted(glob.glob(audio_filefolder+'*.wav'))
audio_durations = [sf.info(each).duration for each in audio_files]
print(f'Total audio recording duration {sum(audio_durations)} s ')

#%%
# There's one extra video TMC file that doesn't match any of the audio files. 
odd_video_file = set(video_durations) - set(audio_durations) 
odd_video_ind = int(np.argwhere(video_durations==list(odd_video_file)[0])[0])
print(f'Odd video file is: {all_csvfiles[odd_video_ind]}')

#%% Preparing audio for first 16s of 22000.TMC 

audio_f22000  = [each for each in audio_files if '1529546136' in each][0]
fs = sf.info(audio_f22000).samplerate
raw_audio, fs = sf.read(audio_f22000, stop=int(16*fs))
#%%
# Get sync channels on both audio interfaces. Channels 0-7 are from device 1
# Channels 8-15 are from device 2. In both devices, there were some channels 
# where the microphones didn't capture much or were dead. 

dev1_channels = [0,1,2,3,4,5]
dev2_channels = [8,9,10,11] # SMP 7, SMP 8 don't really register any signal 

sync_dev1, sync_dev2 = raw_audio[:,7], raw_audio[:,15]
synced_audio = []
start_samples = []
for sync in [sync_dev1, sync_dev2]:
    # segment out the +ve signal parts
    positive_parts, num_parts = ndi.label(sync>0.1)
    # label and identify the frames
    frames = ndi.find_objects(positive_parts)
    # get first sample of the first frame
    video_start = frames[0][0].start
    print(f'Video start: {video_start/fs}')
    start_samples.append(video_start)

    
# The two audio files will be of different lengths, find the shorter one, 
# and clip the longer one to its length

synced_audio = []
for video_sample, dev_channels in zip(start_samples,  [dev1_channels, dev2_channels]):
    synced_audio.append(raw_audio[video_sample:,dev_channels])

num_samples = min([each.shape[0] for each in synced_audio])
synced_audio = np.column_stack([each[:num_samples,:] for each in synced_audio])

#%%
# Verify the sync signal overlay  
plt.figure()
plt.plot(raw_audio[start_samples[0]:192000,7])
plt.plot(raw_audio[start_samples[1]:192000,-1])

#%% 
# And now let's chart out the microphone identities of the channels that were
# kept
#
# | Channel Number | Mic Identity | 
# |----------------|--------------|
# |       0        |    Sanken 9  |
# |       1        |    Sanken 10 |
# |       2        |    Sanken 11 |
# |       3        |    Sanken 12 |
# |       4        |    SMP 1     |
# |       5        |    SMP 2     |
# |       6        |    SMP 3     |
# |       7        |    SMP 4     |
# |       8        |    SMP 5     |
# |       9        |    SMP 6     |

posix_timestamp_wextension = audio_f22000.split('_')[-1]
posix_timestamp = posix_timestamp_wextension.split('.')[0]
sf.write(f'video_synced10channel_first15sec_{posix_timestamp_wextension}',
         synced_audio[:int(fs*15),:], 
         samplerate=fs)

# #%% 
# # Also generate the array geometry file from the TotalStation survey of that day
# # and organise
# survey_points =  pd.read_csv('../acoustic-tracking/2018-06-21/Cave.csv', header=None)
# survey_points.columns = ['point_name','x','y','z']
# to_keep = ['S0', 'S1', 'S2', 'S3', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']
# row_inds = [np.where(survey_points['point_name']==each)[0][0] for each in to_keep]
# micarray_xyz = survey_points.loc[row_inds,:].reset_index(drop=True)

# tristar = micarray_xyz.loc[:3,'x':'z'].to_numpy()
# from scipy.spatial import distance_matrix
# print(distance_matrix(tristar, tristar))

# #%% Switch the handedness of the TotalStation readings - they don't match 
# # what I've been seeing in the real world. 
# micarray_xyz_handedswitch = micarray_xyz.copy()
# micarray_xyz_handedswitch['x'] = micarray_xyz.loc[:,'y']
# micarray_xyz_handedswitch['y'] = micarray_xyz.loc[:,'x']

# micarray_xyz_handedswitch.to_csv(f'arraygeom_2018-06-21_{posix_timestamp}.csv')

# distance_matrix(micarray_xyz_handedswitch.loc[:,'x':'z'], micarray_xyz_handedswitch.loc[:,'x':'z'])

