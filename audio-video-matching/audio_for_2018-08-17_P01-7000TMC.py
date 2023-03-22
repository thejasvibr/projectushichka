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
    
Related module
~~~~~~~~~~~~~~
bat-2dtracks/btrack_checkingout.py
bat-2dtracks/manual_trajmatching.py
bat-2dtracks/traj_correction.py
bat-2dtracks/triangulation_trial.py
"""

#%% Use the previously generated meta-data only files to access # frames in all
# TMC files. 


#%% Get an estimate of the total audio recording time in the first session. 


