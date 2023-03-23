"""
Trying to find the audio recording for 2018-08-17 P01/7000TMC
=============================================================
I've so far managed to get the 3d flight trajectories for 2018-08-17 P01/7000TMC
analysed (first 25 frames).

2018-08-17 is a bit tricky as the GPS-based timestamps aren't there in the TMC files. 
The file creation times have also been lost while transferring from the SD card to disk. 
On top of the lack of file+frame timestamps, there are quite a few recordings with missing
and interrupted frames - leading in frame-fragmentation across multiple files, including
skipped frames etc. 

There are a couple of ways these problems can be handled:

1) use the natural variation in recording times (video files range from 375ish to 377 ish frames)
2) Use the fact that frame numbers are available ina ll TMC frames. The frame number is a rolling
number between 0-8191 that keeps running when the camera is active. 8192 frames is ~5 mins and 28 secs. 
3) Try to account for the fact that audio and video recording time must be equal. 
4) The rolling frame number actually seems to be shared across all three cameras!! The recordings
    start and end at +/- 1 frame difference at most. I'm guessing the frame counter starts from
    the time the 25 Hz signal is given. Since the 25 Hz sync signal is always given simultaneously
    across the cameras -this is a very powerful cue for me to ues!!
5) The timestamp in the audio-file is the END time of the recording. 

Related module
~~~~~~~~~~~~~~
bat-2dtracks/btrack_checkingout.py
bat-2dtracks/manual_trajmatching.py
bat-2dtracks/traj_correction.py
bat-2dtracks/triangulation_trial.py
"""
import glob
import numpy as np
import pandas as pd 
import soundfile as sf

#%% Use the previously generated meta-data only files to access # frames in all
# TMC files. 
all_csvfiles = glob.glob('2018-08-17_video_meta/K1/P000/*.csv') + glob.glob('2018-08-17_video_meta/K1/P0001/*.csv')

total_frames = [pd.read_csv(each, delimiter=';').shape[0] for each in all_csvfiles]


#%% Get an estimate of the total audio recording time in the first session. 

audio_filefolder = 'E:/fieldword_2018_002/actrackdata/wav/2018-08-17_001/'
audio_files_raw = glob.glob(audio_filefolder+'*.wav')
# Remove the last file in the audio folder - as it's from the early morning at ~4 am. 
audio_files = list(filter(lambda X: 'MULTIWAV_2018-08-18_04-35-55_1534556155.WAV' not in X, audio_files_raw))
durations = [sf.info(each).duration for each in audio_files]

print(f'Total audio recording duration {sum(durations)} s ')

#%% Also get the time-gaps between recording ends. First parse the audio-file timestamps out. 
posix_times = np.array([each.split('_')[-1][:-4] for each in audio_files], dtype=np.int64)
end_gaps = np.diff(posix_times)
deriv_gaps = np.diff(end_gaps)

template = np.array([-4, -5, 3, -1])
match_array = []
for i, each in enumerate(deriv_gaps):
    try:
        sub_array = deriv_gaps[i:i+template.size]
        match = np.sqrt(np.sum((template-sub_array)**2))
        match_array.append(match)
    except:
        pass
match_index_start = np.where(np.array(match_array)==0)[0]

# TMC 7000 corresponds to the last index in the derivative of end - gaps which 
# means here it is the 6th file from the pattern's start. The audio durations and 
# frame numbers of the tMC files also match. 

#%%
# | Audio file timestamp | Audio duration (s) |Corresp candidate video file | Video duration(s) |
# |----------------------|--------------------|-----------------------------|-------------------|
# |     1534537235       |        15.04       |            1000.TMC         |     15.04         |
# |     1534537328       |        15.04       |            2000.TMC         |     15.04         |
# |     1534537416       |        15.08       |            3000.TMC         |     15.08         |
# |     1534537507       |        15.0        | 4000+5000 w 7 frames missing| 15.0 with missing |
# |     1534537597       |        15.08       |            6000.TMC         |     15.08         |
# |     1534537686       |        15.04       |            7000.TMC         |     15.04         |
#
# In summary the audio file for 2018-08-17/P01/7000.TMC is the audio with timestamp
# 1534537686 - recorded at 23:28:06. The other corresponding files are also given here. 

#%% Load the matched audio file and check to see how 