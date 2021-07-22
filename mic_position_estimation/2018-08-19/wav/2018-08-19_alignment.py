# -*- coding: utf-8 -*-
"""
Aligning audio for 2018-08-19 speaker playbacks
===============================================
This session shows a different device configuration than those used till now, 
and probably represents the entry of the Fireface 802 and Fireface UC devices. 

28 channels:
    * Channels 0,1,3,4,5: microphones connected to device 1 (channel 2 defunct)
    * Channel 6: Original output signal copy
    * Channel 7: sync signal for device 1
    * Channel 8-11: SANKEN mics on device 1
    * Channel 12-13: blank channels 
    * Channel 14-15: blank channels
    * Channel 16-22: mics connectec to device 2
    * Channel 23: sync signal for device 2
    * Channel 24-25: blank channels
    * Channel 26-27: blank channels

The configuration on 2018-08-17 session reflects the relatively complex mix of
Fireface 802, Fireface UC and the Focusrite Scarlett OctoPre. 


Author: Thejasvi Beleyur
License: Code released with MIT License
"""


import glob
import soundfile as sf
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import scipy.signal as signal 

recording_timestamp = '2018-08-20_09-02-57' 
filenames = glob.glob('./'+'SPKR*'+recording_timestamp+'*')

#%% Load the sync signals only and compare the delay between the two sync channels 
# across various points in time - this will reveal the weird inter-device delay
# that shifts with time. 

fs = 192000

file_info = sf.info(filenames[0])
start_times = np.linspace(2,file_info.duration*0.99,10)
stop_times = start_times+0.04
start_samples = np.int64(fs*start_times)
stop_samples = np.int64(fs*stop_times)

dev2_re_dev1_delay = []

for each_start, each_stop in zip(start_samples, stop_samples):
    partaudio, fs = sf.read(filenames[0], start=each_start, stop=each_stop)
    sync1 = partaudio[:,7]
    sync2 = partaudio[:,23]
    
    part_sync1 = sync1
    part_sync1 *= 1/np.max(part_sync1)
    part_sync2 = sync2
    part_sync2 *= 1/np.max(part_sync2)
    
    cc = signal.correlate(part_sync2, part_sync1, 'full')
    peak = int(np.argmax(cc))
    delay = int(np.round(peak - cc.size*0.5))
    dev2_re_dev1_delay.append(delay)


print(f'Device 2 shows a relative {dev2_re_dev1_delay} samples delay relative to device 1')

#%% As we can see above, the delay ranges are constant through time, which is a good sign!

#%% Load the sync signals only and compare the delay between the two 
partaudio, fs = sf.read(filenames[0], stop=384000)
samples = 240000
sync1 = partaudio[:,7]
sync2 = partaudio[:,23]

part_sync1 = sync1[:samples]
part_sync1 *= 1/np.max(part_sync1)
part_sync2 = sync2[:samples]
part_sync2 *= 1/np.max(part_sync2)

cc = signal.correlate(part_sync2, part_sync1, 'full')
peak = int(np.argmax(cc))
delay = peak - cc.size*0.5

print(f'Device 2 is {delay} samples relative to device 1')


plt.figure()
if delay<0:
    plt.plot(part_sync1[int(abs(delay)):])
else:
    plt.plot(part_sync1[:int(delay)])
plt.plot(part_sync2)

#%% Now make the composite file which needs to be delay adjusted
whole_audio, fs = sf.read(filenames[0])

#%% save file and remove channels without data and sync channels 
with open('../mic2mic_measurements/2018-08-19_goodmics.txt','r') as f:
    goodmic_channels_str = f.readlines()

goodmic_channels_str = goodmic_channels_str[0]
goodmics =  [int(each) for each in goodmic_channels_str.split(',')]

#goodmics += [7,23] #-- to  check that the sync signals are actually getting aligned 

#%% Make an empty array with the [samples x good mics] shape, and shift device 1
# channels to synchronise
shifted_audio = np.zeros((whole_audio.shape[0]+int(delay), len(goodmics)))

for i, each in enumerate(goodmics):
    if np.logical_and(each <=15, delay<0):
        shifted_audio[:,i] = whole_audio[int(abs(delay)):,each]
    elif np.logical_and(each>15, delay<0):
        shifted_audio[:,i] = whole_audio[:int(delay),each]

#%% 
s1 = shifted_audio[-samples:,-1]
s2 = shifted_audio[-samples:,-2]
s1 *= 1/np.max(s1)
s2 *= 1/np.max(s2)
plt.figure()
plt.plot(s1)
plt.plot(s2)

#%%
sf.write('composite_speaker_playback_'+recording_timestamp+'.WAV', 
                                                         shifted_audio, fs)
