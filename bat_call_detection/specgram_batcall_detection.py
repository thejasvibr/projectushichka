# -*- coding: utf-8 -*-
"""
Functions to run the spectrogram based bat  call detection
==========================================================
This module is based on 'detecting_batcalls.py'


Created on Fri Sep 20 13:03:47 2024

@author: theja
"""
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from skimage.filters.rank import entropy
from skimage.morphology import disk, skeletonize
from scipy.ndimage import label, find_objects
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import scipy.signal as signal 

dB = lambda X: 20*np.log10(X)

def bandpass_audio(audio, ba):
    b,a = ba
    return signal.filtfilt(b,a, audio)


def specgram_batcall_detector(audio, fs, **kwargs):
    '''
    Template convolution + entropy map based spectrogram call detector

    Parameters
    ----------
    audio : np.array
    fs : float>0
        sampling rate in Hz.
    
    Keyword arguments
    -----------------
    NFFT, noverlap : int
        Size of the FFT - inputs for plt.specgram. 
        Defaults to 0.5 ms & 0.25 m s respectively. 
    dynamic_range : int
        Maximum dynamic range to consider. Everything below
        max-value - dynamic_range gets suppressed. Defaults
        to 70 dB
    footprint : np.array
        The template of the signal to be detected to generate
        the entropy map. Defaults to 5x5 diagonal matrix.
        Remember that the footprint must be left-right flipped
        to get sharp detections. 
    entropy_threshold : float
        Threshold value below which everything is set to 0. 
        Defaults to 0.5 

    Returns
    -------
    (t, freqs, periodograms) : tuple
        Tuple with the time, frequencies & periodograms of the signal 
    entropy_map : (M,N) np.array
        2D entropy map with no thresholding implemented. 
    call_detections : list with skimage RegionProperties objects
        All call detections as scikit-image regions with relevant
        properties such as bounding boxes etc. 

    '''
    nfft = kwargs.get('NFFT', int(fs*0.5e-3))
    noverlap = kwargs.get('noverlap', int(nfft*0.5))
    # Spectrogram/periodogram generation
    periodograms, freqs, t, img = plt.specgram(audio, Fs=fs, NFFT=nfft,
                                               noverlap=noverlap);
    plt.close()
    # normalise and process spectrogram image
    dynamic_range = kwargs.get('dynamic_range', 70) # dB
    periodograms_proc = periodograms.copy()
    dB_spec = dB(periodograms)
    dB_spec -= dB_spec.max()
    periodograms_proc[dB_spec<=-dynamic_range] = 0

    # Re-scale values to go from 0-1
    periodograms_proc = (periodograms_proc - periodograms_proc.min())/(periodograms_proc.max()-periodograms_proc.min())
    
    footprint = kwargs.get('footprint', np.fliplr(np.eye(5)))
    entropy_map = entropy(periodograms_proc, footprint)
    
    entropy_threshold = kwargs.get('entropy_threshold', 0.5)
    nonzero_entropy = entropy_map > entropy_threshold
    
    
    label_image, num_labels = label(nonzero_entropy, return_num=True)
    call_detections = regionprops(label_image)
    
    return (t, freqs, periodograms), entropy_map, call_detections 

# Plot patches - utility function

def patch_plotter(regions, axis):
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor='red',
            linewidth=1,
        )
        axis.add_patch(rect)
    plt.show()

#%% utility functions that fix indexing issues between bounding region detections
# and 
def limit_inds(indexvalue, array_size):
    ''' limits index values to be >=0 and <= array_size'''
    if indexvalue <= 0:
        output_index = 0
    elif indexvalue >= array_size:
        output_index = array_size-1
    else:
        output_index = indexvalue
    return output_index

def limit_2D_inds(indices, array_dims):
    min_row, min_col, max_row, max_col = indices
    limit_minrow, limit_maxrow = [ limit_inds(each, array_dims[0])  for each in [min_row, max_row]]
    limit_mincol, limit_maxcol = [ limit_inds(each, array_dims[1])  for each in [min_col, max_col]]
    return (limit_minrow, limit_mincol, limit_maxrow, limit_maxcol)




if __name__ == '__main__':
    print('hi')
    import soundfile as sf
    import matplotlib.patches as mpatches

    import os 
        
    audiofile_type = 'single'
    
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
    
    audio_raw, fs = sf.read(audiofile, start=start_sample, stop=stop_sample)
    b,a = signal.butter(1, 9e3/(fs*0.5), 'high')
    audio = np.apply_along_axis(audio_raw, 0, audio_raw, ba=(b,a))
    
    numch = audio.shape[1]

    fig, axs = plt.subplots(numch, 1)
    for chnum in range(numch):
        periodogram_data, entmap, call_detections = specgram_batcall_detector(audio[:,chnum],
                                                                  fs, 
                                                                  dynamic_range=80,
                                                          footprint=np.fliplr(np.eye(10)),
                                                          entropy_threshold=0.7)
        t, freq, periodogram = periodogram_data
        ax = axs[chnum]
        if chnum>0:
            axs[chnum].sharex(axs[0])
        ax.imshow(dB(periodogram), aspect='auto', origin='lower')
        for region in call_detections:
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
    plt.show()
    
            
        
    
    
    