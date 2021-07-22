# -*- coding: utf-8 -*-
"""
Aligning audio for 2018-08-14 speaker playbacks
===============================================
This session shows a different device configuration than those used till now, 
and probably represents the entry of the Fireface 802 and Fireface UC devices. 

28 channels:
    * Channels 0-6: microphones connected to device 1
    * Channel 6: Original output signal copy
    * Channel 7: sync signal for device 1
    * Channel 8-11: SANKEN mics on device 1
    * Channel 12-13: blank channels 
    * Channel 14-15: mic data through ADAT inputs from Focusrite -- can't be time-synchronised!
    * Channel 16-22: mics connectec to device 2
    * Channel 23: sync signal for device 2
    * Channel 24-25: blank channels
    * Channel 26: non-functional mic -- data from ADAT input port through Focusrite digitiser
    * Channel 27: sync signal -- data from ADAT input port through Focusrite digitiser


The configuration on 2018-08-14 session reflects the relatively complex mix of
Fireface 802, Fireface UC and the Focusrite Scarlett OctoPre. 

Difficulty with ADAT channels
-----------------------------
The ADAT channels of both the 802 and UC were used. The ADC in the OctoPre
was used to output ADAT data into the 802 and UC. However using the ADAT to record
audio means having to account for time-delays in digitisation that occured at the
OctoPre. Looking at the raw audio recording, it's pretty clear that there is a 
non-negligible digitisation delay between the OctoPre and the UC for instance -- which 
means I'll have to anyway remove these channels sadly - an overall loss of 3 mics from the 
20 mics!

Another additional issue to keep in mind is that the dynamic range of the OctoPre isn't known
and this in turn needs to accounted for. Overall - it'll be too much effort I think, and thus
not worth including the ADAT channels, even though so much time was spent incorporating them
into the recording system. 


Author: Thejasvi Beleyur
License: Code released with MIT License
"""
import glob
import soundfile as sf
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import scipy.signal as signal 


recording_timestamp = '2018-08-15_07-47-10' 
filenames = glob.glob('./'+'SPKR*'+recording_timestamp+'*')

#%% Load the sync signals only and compare the delay between the two 
partaudio, fs = sf.read(filenames[0], stop=384000)
samples = 240000
sync1 = partaudio[:,7]
sync2 = partaudio[:,23]

#%% 
plt.figure()
plt.plot(sync1[:samples], label='device 1')
plt.plot(sync2[:samples], label='device 2')
plt.legend()
#%% 
part_sync1 = sync1[:samples]
part_sync1 *= 1/np.max(part_sync1)
part_sync2 = sync2[:samples]
part_sync2 *= 1/np.max(part_sync2)

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

with open('../mic2mic_measurements/2018-08-14_goodmics.txt','r') as f:
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




