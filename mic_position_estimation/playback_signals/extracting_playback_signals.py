# -*- coding: utf-8 -*-
"""
Wrangling the original playback signals
=======================================
This module documents how I 'restored' the playback signals into a form 
that can be used for the constant-offset version of the Structure-From-Sound


All the original signals are hosted in the 'AV_calibration_playback' repo
here: https://github.com/thejasvibr/AV_calibration_playback.git


Types of playback signals
-------------------------
Over the course of the field season there were 3 types of playback signals:
    * upward hyperbolic sweep 8ms duration, 1Hz playback rate
        
        Sessions used: 2018-06-19, 2018-06-21, 2018-06-22, 
        
    * upward hyperbolic sweep 8ms duration, 2Hz playback rate
        
    Sessions used: 2018-07-14

    * 9chirp series. 9 signals in 200 ms long units, followed by 200 ms x 9 units silence
        The 9 signals are a series of three signal types X three durations 
        The signal types are linear sweep, logarithmic sweep and bidirectional linear sweep.
        The durations are 6, 12 and 24 ms
        
        Sessions used: 2018-07-21, 2018-07-25, 2018-07-28, 2018-08-14,2018-08-17, 2018-08-19


Author: Thejasvi Beleyur 
License : MIT
"""

import glob
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
plt.rcParams['agg.path.chunksize'] = 10000
import soundfile as sf

fs = 192000

#%%
local_repo_path = '../../../../common/Python_common/AV_calibration_playback/'
playback_files = glob.glob(local_repo_path+'*.npy')
playbacks = [ np.load(each) for each in playback_files]

#%% The hyperbolic sweep at 1 and Hz rate:
sweep, fs = sf.read(local_repo_path+'chirp.WAV')

sweep_1Hz = np.concatenate((np.zeros(fs-sweep.size), sweep))

sweep_2Hz = np.concatenate((np.zeros(96000-sweep.size), sweep))



plt.figure()
a0 = plt.subplot(211)
plt.specgram(sweep_1Hz+np.random.normal(0,1e-5,sweep_1Hz.size), Fs=fs)
plt.subplot(212, sharex=a0)
plt.specgram(sweep_2Hz+np.random.normal(0,1e-5,sweep_2Hz.size), Fs=fs)

#%% Saving the sweeps
signal_files = ['hyperbolicsweep_1Hz.wav', 'hyperbolicsweep_2Hz.wav', '9chirp_set.wav']


sf.write(signal_files[0], sweep_1Hz, fs)
sf.write(signal_files[1], sweep_2Hz, fs)


#%% The 9chirp series 
multichirp_1 = playbacks[1].flatten()

sf.write(playback_files[2], multichirp_1, fs)
#%% 
plt.figure()
plt.specgram(multichirp_1, Fs=fs)

#%% Write the sessions and file names 
data =     {'2018-06-19': signal_files[0],
            '2018-06-21': signal_files[0],
            '2018-06-22': signal_files[0],
            '2018-07-14': signal_files[1],
            '2018-07-21': signal_files[2],
            '2018-07-25': signal_files[2],
            '2018-07-28': signal_files[2],
            '2018-08-14': signal_files[2],
            '2018-08-17': signal_files[2],
            '2018-08-19': signal_files[2]}
keys = []
entries = []
for key, entry in data.items():
    keys.append(key)
    entries.append(entry)
    
df = pd.DataFrame(data={'session_date':keys, 'playback_files':entries})
df.to_csv('session_wise_signalfiles.csv')
