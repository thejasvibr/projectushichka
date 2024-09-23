# -*- coding: utf-8 -*-
"""
Applying the bat call detector on each channel individually - MULTIBAT
======================================================================

Looks like there is a need to finetune the parameters for each channel - and 
so here we go. 


Some insights on good parameters
--------------------------------
* A smaller footprint is often better. e.g. np.eye(5) is better than np.eye(10), 
  especially when there are overlaps because of echoes. A footprint that is too 
  small can cuase the opposite problems in fragmenting the same call into different
  calls. 
* Entropy threshold. In general you want to have it at a 'reasonable' value (e.g. 0.5). 
  Especially when you notice the direct call + reverb are getting fused in the segmentation, 
  then the entropy threshold can be increased to improve separation. 

Created on Mon Sep 23 10:08:58 2024

@author: theja
"""
import matplotlib.pyplot as plt
import numpy as np 
import os 
import pandas as pd
import scipy.signal as signal 
import soundfile as sf
from specgram_batcall_detection import *

#%%
# Load audio and select video-tracking specific snippets
audiofile_type = 'multi'

if audiofile_type == 'single':
    audiofile = os.path.join('1529546136_input',
                                  'video_synced10channel_first15sec_1529546136.wav')
elif audiofile_type == 'multi':
    audiofile = os.path.join('1529543496_input',
                                      'video_synced10channel_first15sec_1529543496.wav')

fs = sf.info(audiofile).samplerate


if '6136' in audiofile:
    t_start, t_stop = 0, 1
    start_sample, stop_sample = int(fs*t_start), int(fs*t_stop)
elif '3496' in audiofile:
    t_start, t_stop = 12.4, 13.4
    start_sample, stop_sample = int(fs*t_start), int(fs*t_stop)

#%% Load and bandpass the audio snippets
audio_raw, fs = sf.read(audiofile, start=start_sample, stop=stop_sample)
b,a = signal.butter(1, np.array([9e3, 90e3])/(fs*0.5), 'band')
audio = np.apply_along_axis(bandpass_audio, 0, audio_raw, ba=(b,a))

#%% Channel-wise parameters 
common_footprint = np.fliplr(np.eye(4))
parameters_by_channel = {}
default_channel_params = {'dynamic_range':70, 
                        'footprint':common_footprint,
                        'entropy_threshold': 0.5,
                        'NFFT' : int(fs*0.5e-3), 
                        'noverlap':int(fs*0.25e-3)}
# Some mics have different input parameters
for channel_num in range(10):
    parameters_by_channel[channel_num] = default_channel_params
    if channel_num in [4,5,8,]:
        parameters_by_channel[channel_num]['entropy_threshold'] = 1.0

wavfilename = os.path.split(audiofile)[-1].split('.')[0]

all_channel_detections = []
for channel_num in range(audio.shape[1]):
    periodogram_data, entmap, call_detections = specgram_batcall_detector(audio[:,channel_num],
                                                              fs, 
                                                              **parameters_by_channel[channel_num])
    
    
    colormap = 'viridis'
    
    fig, axs = plt.subplots(3,1, figsize=(18,10))
    ax = axs[0]
    t, freq, periodogram = periodogram_data
    ax.imshow(dB(periodogram), aspect='auto', origin='lower',
              cmap=colormap, )
    patch_plotter(call_detections, ax)
    ax.set_title(f'Channel number: {channel_num} - raw detections')
    ax.set_ylabel(f'Frequency, 1 unit = {np.around(freq[1]-freq[0], 1)*1e-3} kHz')
    
    axz = axs[1]
    axz.sharex(ax)
    t, freq, periodogram = periodogram_data
    axz.imshow(dB(periodogram), aspect='auto', origin='lower',
              cmap=colormap, )
    axz.set_title(f'Duration + bandwidth restricted detections')
    restricted_calldets = []
    call_detections_truescale = []
    
    for each in call_detections:
        f_start, t_start, f_stop, t_stop = limit_2D_inds(each.bbox, periodogram.shape)
        durn = t[t_stop] - t[t_start]
        bandwidth = freq[f_stop] - freq[f_start]
        
        sensible_durn = np.logical_and(durn>=1.5e-3, durn<=7e-3)
        sensible_spectralrange = np.logical_and(bandwidth>=9e3, freq[f_start]<=40e3)
        
        if np.logical_and(sensible_durn, sensible_spectralrange):
            restricted_calldets.append(each)
            call_detections_truescale.append((t[t_start], t[t_stop], freq[f_start], freq[f_stop]))
    
    patch_plotter(restricted_calldets, axz)
    
    axx = axs[2]
    axx.sharex(ax)
    axx.imshow(entmap, origin='lower', aspect='auto', cmap=colormap,
               )
    axx.set_title('Entropy map - raw')
    axx.set_xlabel(f'Time, 1 unit = {np.around(t[1]-t[0], 5)*1e3} ms')
    plt.savefig(f'{wavfilename}_channel_num_{channel_num}.png')
    
    #%%
    # Format the restricted detections into a dataframe -> csv file. 
    call_detections_df = pd.DataFrame(data=None,
                                      columns=['audio_file', 'channel_num', 
                                               't_start', 't_stop', 'freq_start',
                                             'freq_stop', 
                                             'snippet_start', 'snippet_stop'])
    start_times = list(map(lambda X: X[0], call_detections_truescale))
    stop_times = list(map(lambda X: X[1], call_detections_truescale))
    start_freq = list(map(lambda X: X[2], call_detections_truescale))
    stop_freq = list(map(lambda X: X[3], call_detections_truescale))
    
    call_detections_df['t_start'] = start_times
    call_detections_df['t_stop']  = stop_times
    call_detections_df['freq_start'] = start_freq
    call_detections_df['freq_stop'] = stop_freq
    call_detections_df['snippet_start'] = t_start
    call_detections_df['snippet_stop'] = t_stop
    call_detections_df['audio_file'] = os.path.split(audiofile)[-1]
    call_detections_df['channel_num'] = channel_num
    call_detections_df['filesnippter_tstart'] = t_start
    call_detections_df['filesnippter_tstop'] = t_stop
    all_channel_detections.append(call_detections_df)
multibat_allchannel = pd.concat(all_channel_detections).to_csv(f'{wavfilename}_batcall-detections.csv')

#%%
# Now sanity test these detections -- replot all the saved  detections just to make sure it's okay. 

import glob
from dataclasses import dataclass
 
@dataclass
class custom_region():
    """A class that mocks a skimage region object"""
    bbox : tuple
    
csvfiles = glob.glob('*.csv')
detection_data = pd.read_csv(csvfiles[0])
bychannel = detection_data.groupby('channel_num')

chnum = 5
pgram, f, t, img  = plt.specgram(audio[:,chnum], Fs=fs, NFFT=192, noverlap=96);plt.close()
fig, ax00 = plt.subplots(1,1)
ax00.imshow(dB(pgram), origin='lower', aspect='auto', extent=[0, t[-1], f[0], f[-1]])
ax00.set_xlim(0,1)
ax00.set_ylim(0,96000)
dets = []
for i, row in bychannel.get_group(chnum).iterrows():
    aa = custom_region((row['t_start'], row['freq_start'], row['t_stop'], row['freq_stop']))
    dets.append(aa)
#patch_plotter(dets, ax00)
import matplotlib.patches as mpatches

for region in dets:
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle(
        (minr, minc),
        maxr - minr,
        maxc - minc,
        fill=False,
        edgecolor='red',
        linewidth=1,
    )
    ax00.add_patch(rect)
plt.show()












