# -*- coding: utf-8 -*-
"""
Detecting bat calls 
===================


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
ch_num = 5
periodograms, freqs, t, img = plt.specgram(audio[:,ch_num], Fs=fs, NFFT=96, noverlap=48)

#%% We're getting somewhere - btut  the median filtering is actually removing some info... 
# What about entropy filtering? 
# Let's first try 2D entropy filtering 

dynamic_range = 70 # dB
periodograms_proc = periodograms.copy()
dB_spec = dB(periodograms)
dB_spec -= dB_spec.max()

periodograms_proc[dB_spec<=-dynamic_range] = 0

#%%

# Re-scale values to go from 0-1
periodograms_proc = (periodograms_proc - periodograms_proc.min())/(periodograms_proc.max()-periodograms_proc.min())

diagonal_template = np.fliplr(np.eye(5))
templ = entropy(periodograms_proc, diagonal_template)

#%% 
# Segment out regions that are not 0.
nonzero_entropy = templ > 0.5

label_image, num_labels = label(nonzero_entropy, return_num=True)
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

durn_threshold = 1.5e-3 # s 
minfreq_thresh = 30e3 # Hz
bw_thresh = 20e3 # Hz

good_detection = []
inds = np.zeros((num_labels, 4))
for i, region in enumerate(regionprops(label_image)):
    minr, minc, maxr, maxc = region.bbox
    inds[i,:] = region.bbox
    
    # minfreq, t_start, maxfreq, t_stop = freqs[minr], t[minc], freqs[maxr], t[maxc]
    
    # duration = t_stop - t_start
    # bandwidth = maxfreq - minfreq
    
    # # first filter according to duration  
    # longenough = duration >= durn_threshold
    # # then filter by min freq
    # minfreq_lowenough = minfreq <= minfreq_thresh
    # # and is there sufficient bandwidth?
    # bandwidth_enough = bandwidth >= bw_thresh
    
    # if np.all([longenough, minfreq_thresh, bandwidth_enough]):
    #     good_detection.append(region)

#%% Now visualise this and compare with all regions 
num_plots = 211
plt.figure()
ax0 = plt.subplot(num_plots)

ax0.imshow(image_label_overlay, aspect='auto', origin='lower',)

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
    ax0.add_patch(rect)
    
textx, texty = 1500, 45    
plt.text(textx, texty, 'Detections', color='w')
ax0.set_yticks(np.arange(image_label_overlay.shape[0])[::5], freqs[::5]*1e-3)
ax0.set_xticks(np.arange(image_label_overlay.shape[1])[::125],
               np.round(t[::125],2))
plt.tight_layout()
plt.grid()
plt.show()

ax1 = plt.subplot(num_plots+1, sharex=ax0)
ax1.imshow(image_label_overlay, aspect='auto', origin='lower')

for region in good_detection:
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
    ax1.add_patch(rect)
ax1.set_yticks(np.arange(image_label_overlay.shape[0])[::10], freqs[::10]*1e-3)


