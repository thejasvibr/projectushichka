# -*- coding: utf-8 -*-
"""
Detecting bat calls
===================

Some ideas to try out: 
    
    > median subtract from frequency bands
    > hand

Created on Mon Sep  2 14:38:47 2024


@author: theja
"""
import soundfile as sf
import numpy as np 
import scipy.signal as signal 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import glob
import os 
from skimage.filters.rank import entropy
from skimage.morphology import disk, skeletonize
from scipy.ndimage import label, find_objects
from skimage.measure import label, regionprops
from skimage.color import label2rgb
dB = lambda X: 20*np.log10(abs(X))

audiofile_type = 'multi'

if audiofile_type == 'single':
    audiofile = os.path.join('1529546136_input',
                                  'video_synced10channel_first15sec_1529546136.wav')
elif audiofile_type == 'multi':
    audiofile = os.path.join('1529543496_input',
                                      'video_synced10channel_first15sec_1529543496.wav')

fs = sf.info(audiofile).samplerate


if '6136' in audiofile:
    start_sample, stop_sample = int(fs*0), int(fs*1)
elif '3496' in audiofile:
    start_sample, stop_sample = int(fs*12.4), int(fs*13.4)

audio, fs = sf.read(audiofile, start=start_sample, stop=stop_sample)
#%%
ch_num = 7
periodograms, freqs, t, img = plt.specgram(audio[:,ch_num], Fs=fs, NFFT=96, noverlap=48)
# plt.figure()
# aa = plt.subplot(311)

# where_highfreqs = np.logical_and(freqs>= 50e3, freqs<=95e3)
# highfreqs = freqs[where_highfreqs]
# highfreq_periodograms = periodograms[where_highfreqs, :]

# sum_highfreq_pgrams = highfreq_periodograms.sum(axis=0)
# smoothed_highfreqs = signal.medfilt(dB(sum_highfreq_pgrams), 15)

# plt.subplot(312, sharex=aa)
# row_ind = -1
# plt.plot(t, dB(sum_highfreq_pgrams))

# plt.subplot(313, sharex=aa)
# plt.plot(t, smoothed_highfreqs)
#%%
# Fitting an overall polynomial is not so useful...
# pp0 = np.polynomial.Polynomial.fit(t, dB(sum_highfreq_pgrams), deg=100)
# plt.figure()
# ab = plt.subplot(311)
# plt.plot(t, pp0(t))
# plt.plot(t, dB(sum_highfreq_pgrams))
# plt.subplot(312, sharex=ab)
# plt.plot(t, dB(sum_highfreq_pgrams)-pp0(t))
# plt.subplot(313, sharex=ab)
# plt.specgram(audio[:,ch_num], Fs=fs, NFFT=96, noverlap=48);

#%%
# What about splitting the signal into parts and then applying short-length detrending
# def make_waveformplot(X, fs):
#     t_x = np.linspace(0, X.size/fs, X.size)
#     plt.plot(t_x, X)

# num_parts = 100
# parts_highfreq = np.array_split(dB(sum_highfreq_pgrams), num_parts)
# parts_t_highfreq = np.array_split(t, num_parts)

# polynom_parts = []
# detrended = []
# for (t_each, each) in zip(parts_t_highfreq, parts_highfreq):
#     polynom = np.polynomial.Polynomial.fit(t_each, each, deg=2)
#     polynom_pred = polynom(t_each)
#     polynom_parts.append(polynom_pred)
#     detrended.append(each - polynom_pred)

# all_detrended = np.concatenate(detrended)
# plt.figure()
# bb = plt.subplot(311)
# plt.specgram(audio[:,ch_num], Fs=fs, NFFT=96, noverlap=48)
# plt.subplot(312, sharex=bb)
# plt.plot(t, all_detrended)
# plt.subplot(313, sharex=bb)
# medfilt_detrended = signal.medfilt(all_detrended, 7)
# plt.plot(t, medfilt_detrended)

#%% We're getting somewhere - btut  the median filtering is actually removing some info... 
# What about entropy filtering? 
# Let's first try 2D entropy filtering 

dynamic_range = 70 # dB
periodograms_proc = periodograms.copy()
dB_spec = dB(periodograms)
dB_spec -= dB_spec.max()

periodograms_proc[dB_spec<=-dynamic_range] = 0

#%%
#raise NotImplementedError('Dynamic range restrictionot yet applied. ')

# periodograms_proc_colmeans = np.tile(np.mean(periodograms_proc, axis=1),
#                                      periodograms_proc.shape[1]).reshape(periodograms_proc.shape)
periodograms_proc = (periodograms_proc - periodograms_proc.min())/(periodograms_proc.max()-periodograms_proc.min())

diagonal_template = np.fliplr(np.eye(5))
templ = entropy(periodograms_proc, diagonal_template)

#%% 
# Segment out regions that are not 0.
nonzero_entropy = templ > 0.5

label_image = label(nonzero_entropy)
image_label_overlay = label2rgb(label_image, image=templ, bg_label=0)

num_plots = 411
plt.figure()
ax = plt.subplot(num_plots+3)

ax.imshow(image_label_overlay, aspect='auto', origin='lower')

for region in regionprops(label_image):
    # take regions with large enough areas
    #if region.area >= 0.5:
    # draw rectangle around segmented coins
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle(
        (minc, minr),
        maxc - minc,
        maxr - minr,
        fill=False,
        edgecolor='red',
        linewidth=2,
    )
    ax.add_patch(rect)
textx, texty = 1500, 45    
plt.text(textx, texty, 'Detections', color='w')
ax.set_axis_off()
plt.tight_layout()
plt.show()

plt.subplot(num_plots, sharex=ax)
plt.imshow(dB(periodograms), aspect='auto', origin='lower')

plt.subplot(num_plots+1, sharex=ax)
plt.imshow(dB(periodograms_proc+1e-12), aspect='auto', origin='lower')
plt.text(textx, texty, 'thresholded spectrogram', color='w')
plt.subplot( num_plots+2, sharex=ax)
plt.imshow(templ, aspect='auto', origin='lower')
plt.text(textx, texty, 'entropy map', color='w')

#%%
# What if you also do some further skeletonising on the entropy map?
skelet_entropy = skeletonize(nonzero_entropy)

sk_label_image = label(skelet_entropy)
sk_image_label_overlay = label2rgb(sk_label_image, image=skelet_entropy, bg_label=0)

plt.figure()
axsk = plt.subplot(num_plots+3)

axsk.imshow(sk_image_label_overlay, aspect='auto', origin='lower')

for region in regionprops(sk_label_image):
    # take regions with large enough areas
    #if region.area >= 0.5:
    # draw rectangle around segmented coins
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle(
        (minc, minr),
        maxc - minc,
        maxr - minr,
        fill=False,
        edgecolor='red',
        linewidth=2,
    )
    axsk.add_patch(rect)

textx, texty = 1500, 45    
plt.text(textx, texty, 'Detections', color='w')
axsk.set_axis_off()
plt.tight_layout()
plt.show()

plt.subplot(num_plots, sharex=axsk)
plt.imshow(dB(periodograms), aspect='auto', origin='lower')

plt.subplot(num_plots+1, sharex=axsk)
plt.imshow(dB(periodograms_proc+1e-12), aspect='auto', origin='lower')
plt.text(textx, texty, 'thresholded spectrogram', color='w')
plt.subplot( num_plots+2, sharex=axsk)
plt.imshow(skelet_entropy, aspect='auto', origin='lower')
plt.text(textx, texty, 'skeletonised entropy map', color='w')



#%%
# Now also do some filtering on the detected regions themselves - they shouldn't be 
# only high-frequency regions, and also include some low-frequency regions. 

all_detections = regionprops(label_image)


