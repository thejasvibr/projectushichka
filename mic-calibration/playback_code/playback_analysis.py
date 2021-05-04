# -*- coding: utf-8 -*-
"""
Module to split and analyse the smicrophone calibration data
Created on Tue Apr 27 12:37:00 2021

@author: tbeleyur
"""

import numpy as np 
import pandas as pd
import tqdm
import os
import scipy.signal as signal 
import scipy.io.wavfile as WAV
#import matplotlib.pyplot as plt
#plt.rcParams['agg.path.chunksize'] = 100000
# import soundfile as sf

dB = lambda X: 20*np.log10(abs(X))

def wave_p2p(X):
    '''
    '''
    return np.max(X)- np.min(X)

def wave_peak(X):
    return np.max(np.abs(X))
    
def peak_frequency(X,fs=192000):
    x_fft = np.fft.rfft(X)
    highest_power_index = np.argmax(x_fft)
    x_freqs = np.fft.rfftfreq(X.size, 1.0/fs)
    peak_freq = x_freqs[highest_power_index]
    return peak_freq

def power_spectrum(X, fs=192000):
    '''
    Returns
    -------
    x_freqs, power
    '''
    x_fft = np.fft.rfft(X)
    power = dB(x_fft)
    x_freqs = np.fft.rfftfreq(X.size, 1.0/fs)
    return x_freqs, power
    

def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    Code courtesy: user 'endolith'  https://gist.github.com/endolith/1257010
    """
    return np.sqrt(np.mean(np.abs(a)**2))


def rms_fft(spectrum):
    """
    Use Parseval's theorem to find the RMS value of a signal from its fft,
    without wasting time doing an inverse FFT.
    For a signal x, these should produce the same result, to within numerical
    accuracy:
    rms_flat(x) ~= rms_fft(fft(x))
    
    Code courtesy : user 'endolith'  https://gist.github.com/endolith/1257010
    """
    return rms_flat(spectrum)/np.sqrt(len(spectrum))


def extract_bandrms_from_fft(bandwidth,spectrum, freqs):
    '''
    '''
    absfreq = np.abs(freqs)
    within_band = np.logical_and(absfreq>=bandwidth[0], absfreq<=bandwidth[1])
    not_in_band = np.invert(within_band)
    fft_in_band = spectrum.copy()
    fft_in_band[not_in_band] = 0
    return rms_fft(fft_in_band)
    return within_band

def extract_inband_rms(freq_band, audio, fs):
    '''
    Calculates the RMS contribution of a given frequency band
    
    Parameters
    ----------
    audio:  np.array
    fs : float. 
        Sampling rate, Hz
    freq_band: list-like
        2 entries with lower and upper frequencies in Hz. 
    
    Returns 
    -------
    band_rms : float 
    '''
    audio_fft = np.fft.fft(audio)
    audio_freqs = np.fft.fftfreq(audio.size, 1.0/fs)
    band_rms = extract_bandrms_from_fft(freq_band, audio_fft, audio_freqs)
    return band_rms
    
def get_only_tones(audio, tones_from=1.4, fs=192000):
    only_tones = audio[int(fs*tones_from):]
    return only_tones



def get_tones_rms(audio, tone_freqs, fs=192000, bandwidth=500):
    '''
    '''
    tones_rms = []
    signal_fft = np.fft.fft(audio)
    fftfreqs = np.fft.fftfreq(audio.size, 1.0/fs)
    
    for peak_freq in tqdm.tqdm(tone_freqs):
        if np.logical_or(peak_freq<=0, peak_freq>=fs*0.5):
            raise ValueError(f'A peakfrequency <0 or > Nyquist frequency ({peak_freq}Hz) has been detected. Aborting....')
        signal_band = np.array([peak_freq-bandwidth, peak_freq+bandwidth])
        signal_band[signal_band<0] = 0
        signal_band[signal_band>fs*0.5] = fs*0.5
        rms_value = extract_bandrms_from_fft(signal_band, signal_fft, fftfreqs)
        tones_rms.append(rms_value)
    return pd.DataFrame(data={'tone_freqs':tone_freqs, 'tone_rms': tones_rms})
        
 