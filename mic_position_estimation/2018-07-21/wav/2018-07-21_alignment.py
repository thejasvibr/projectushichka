# -*- coding: utf-8 -*-
"""
Aligning audio for 2018-07-21 speaker playbacks
===============================================
This session shows a different device configuration than those used till now, 
and probably represents the entry of the Fireface 802 and Fireface UC devices. 

16 channels:
    * Channels 0-5: microphones connected to device 1
    * Channel 6: Original output signal copy
    * Channel 7: sync signal for device 1
    * Channel 8-13: microphones connected to device 2
    * Channel 14: empy channel
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


recording_timestamp = '2018-07-22_07-20-02' # 2 mins long, 16 audio files at 192 kHz
filenames = glob.glob('./'+'SPKR*'+recording_timestamp+'*')

#%% Load the sync signals only and compare the delay between the two 
partaudio, fs = sf.read(filenames[0], stop=384000)
samples = 240000
sync1 = partaudio[:,7]
sync2 = partaudio[:,15]

#%% 
plt.figure()
plt.plot(sync1[:samples], label='device 1')
plt.plot(sync2[:samples], label='device 2')
plt.legend()
#%% 
part_sync1 = sync1[:samples]
part_sync2 = sync2[:samples]

cc = signal.correlate(part_sync2, part_sync1, 'full')
peak = int(np.argmax(cc))
delay = peak - cc.size*0.5

print(f'Device 2 is {delay} samples relative to device 1')
#%%
plt.figure()
plt.plot(cc)
plt.plot(peak, cc[peak],'*')
plt.vlines(part_sync2.size,0,np.max(cc))

#%%
plt.figure()
if delay<0:
    plt.plot(part_sync1[int(abs(delay)):])
else:
    plt.plot(part_sync1[:int(delay)])
plt.plot(part_sync2)

#%% Now make the composite file which needs to be delay adjusted
whole_audio, fs = sf.read(filenames[0])

#%% save file and remove channels without data and sync channels 

with open('../mic2mic_measurements/2018-07-21_goodmics.txt','r') as f:
    goodmic_channels_str = f.readlines()

goodmic_channels_str = goodmic_channels_str[0]
goodmics =  [int(each) for each in goodmic_channels_str.split(',')]

# goodmics += [7,15] #-- to  check that the sync signals are actually getting aligned 

#%% Make an empty array with the [samples x good mics] shape, and shift device 1
# channels to synchronise
shifted_audio = np.zeros((whole_audio.shape[0]+int(delay), len(goodmics)))

for i, each in enumerate(goodmics):
    if np.logical_and(each <=11, delay<0):
        shifted_audio[:,i] = whole_audio[int(abs(delay)):,each]
    elif np.logical_and(each>11, delay<0):
        shifted_audio[:,i] = whole_audio[:int(delay),each]

#%% 
plt.figure()
plt.plot(shifted_audio[-samples:,-1])
plt.plot(shifted_audio[-samples:,-2])

#%%
sf.write('composite_speaker_playback_'+recording_timestamp+'.WAV', 
                                                         shifted_audio, fs)

#%% For 2018-07-14, we also have the actual output signal which was played back
# through the speaker. 
output_signal = whole_audio[int(abs(delay)):,6]
sf.write('digital_playbacksignal_'+recording_timestamp+'.WAV', output_signal, fs)




