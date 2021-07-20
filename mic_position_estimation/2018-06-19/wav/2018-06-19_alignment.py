# -*- coding: utf-8 -*-
"""
Aligning audio for 2018-06-19 speaker playbacks
===============================================

16 files for 16 channels:
    * Channels 0-5 : microphones connected to device 1
    * Channel 6: no mic
    * Channel 7: sync signal for device 1
    * Channel 8-12: microphones connected to device 2
    * Channel 13-14: defective mic/ no mic
    * Channel 15: sync signal for device 2

Author: Thejasvi Beleyur
License: Code released with MIT License
"""
import glob
import soundfile as sf
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import scipy.signal as signal 


recording_timestamp = '1529465474' # 2 mins long, 16 audio files at 192 kHz
filenames = glob.glob('./'+'*'+recording_timestamp+'*')
sync1_file = list(filter(lambda X: 'Mic07' in X, filenames))[0]
sync2_file = list(filter(lambda X: 'Mic15' in X, filenames))[0]

#%% Load the sync signals only and compare the delay between the two 
sync1, fs = sf.read(sync1_file)
sync2, fs = sf.read(sync2_file)


#%% 
half_cycle = int(0.7*0.04*fs)
plt.figure()
plt.plot(sync1[:half_cycle], label='device 1')
plt.plot(sync2[:half_cycle], label='device 2')
plt.legend()
#%% 
samples = int((1/25)*fs*0.7)
part_sync1 = sync1[:samples]
part_sync2 = sync2[:samples]

cc = signal.correlate(part_sync2, part_sync1, 'full')
peak = int(np.argmax(cc))
delay = peak - cc.size*0.5

print(f'Device 2 is {delay} samples ahead of device 1')
#%%
plt.figure()
plt.plot(cc)
plt.plot(peak, cc[peak],'*')
plt.vlines(part_sync2.size,0,np.max(cc))


#%% Now make the composite file which doesn't need to be delay adjusted - just < 1 sample difference. 
multichannel_audio = np.zeros((sync1.size,16))

all_audio = []
for i,each in enumerate(filenames):
    audio, fs = sf.read(each)
    multichannel_audio[:,i] = audio
#%% save file and remove channels without data and sync channels 
goodmic_channels = [0,1,2,3,4,5,8,9,10,11,12]

sf.write('composite_speaker_playback_'+recording_timestamp+'.WAV', 
         multichannel_audio[:,goodmic_channels], fs)





