"""
Module holding all relevant analysis functions for the calbration playback
that TB has been using over May-June 2021. 

The calibration playback consists of 5 ultrasonic sweeps and 85 tones between 10-94kHz in 1kHz steps. 

- Thejasvi Beleyur
This code is released under an MIT License. 
"""
import numpy as np 
import pandas as pd
import scipy.signal as signal 
import sys
sys.path.append('../')
sys.path.append('../../')
import playback_code.playback_analysis as pa
from fullscale_calculations import * 
def get_loudest_part(X, fs=192000, winsize=0.025, threshold=20):
    x_sq = np.sqrt(np.abs(X)**2)
    winsamples = int(fs*winsize)
    running_mean = np.convolve(x_sq, np.ones(winsamples)/winsamples, mode='same')
    #threshold = np.percentile(running_mean, windowlimit)
    db_runningmean = 20*np.log10(running_mean)
    indices =  db_runningmean >= np.max(db_runningmean)-threshold
    return indices, running_mean

def calculate_tones_rec_level(audio, fs, tone_freqs, FS_dbu, gain, sensitivity_dbvrms, freq_bw=500):
    '''
    '''
    
    audio_parts = np.array_split(audio, len(tone_freqs))
    
    tones_rms = []
    for peak_freq, tone_part in zip(tone_freqs, audio_parts):
        loudest_inds, _ = get_loudest_part(tone_part, fs=fs)
        audio_part = tone_part[loudest_inds]
        signal_fft = np.fft.fft(audio_part)
        fftfreqs = np.fft.fftfreq(audio_part.size, 1/fs)
        if np.logical_or(peak_freq<=0, peak_freq>=fs*0.5):
            raise ValueError(f'A peakfrequency <0 or > Nyquist frequency ({peak_freq}Hz) has been detected. Aborting....')
        signal_band = np.array([peak_freq-freq_bw, peak_freq+freq_bw])
        signal_band[signal_band<0] = 0
        signal_band[signal_band>fs*0.5] = fs*0.5
        rms_value = pa.extract_bandrms_from_fft(signal_band, signal_fft, fftfreqs)
        tones_rms.append(rms_value)
    
    #tones_rms = pa.get_tones_rms(audio, tone_freqs,fs=fs,bandwidth=freq_bw)
    tones_rms = pd.DataFrame(data={'tone_rms':tones_rms})
    max_rms = 1/(np.sqrt(2)) # for Vp=1 Vrms = (1/sqrt(2))*Vp
    tones_rms['dbrms_re_max'] = pa.dB(tones_rms['tone_rms']/max_rms)
    tones_rms['dbrms_wogain'] = tones_rms['dbrms_re_max']-gain
    tones_rms['db_FS'] = pa.dB(dbu2vrms(FS_dbu))
    tones_rms['tone_db'] = tones_rms['db_FS'] + tones_rms['dbrms_wogain']
    tones_rms['tone_freqs'] = tone_freqs
    sensitivity_vrms = 10**(sensitivity_dbvrms/20.0) # Vrms/Pa
    ref = 20*10**-6 # Pa

    tones_rms['Pa_rms'] = 10**(tones_rms['tone_db']/20.0)*(1/sensitivity_vrms)
    tones_rms['dbspl_rms'] = pa.dB(tones_rms['Pa_rms']/ref)
    return tones_rms


def make_avged_fft(recording_name, fft_dictionary):
    # first check all ffts are the same size 
    fft_sizes = [len(fft_dictionary[recording_name][i][:,1]) for i in range(5)]
    if not np.all(np.array(fft_sizes)==fft_sizes[0]):
        raise ValueError(f'all FFTs not same size: {fft_sizes}')
    all_ffts = np.array([fft_dictionary[recording_name][i][:,1] for i in range(5)]).reshape(5,-1)
    avg_fft = 20*np.log10(np.mean(10**(all_ffts/20.0), axis=0))
    freqs = fft_dictionary[recording_name][0][:,0]
    return avg_fft, freqs

def calculate_average_sweep_spectrum(audio, fs):
    sweeps = np.array_split(audio[int(fs*0.5):int(fs*1.5)],5)

    sweep_ffts = {}

    for i,each in enumerate(sweeps):
        sweeps_bp = signal.lfilter(b,a,each)
        sweepregion, _ = get_loudest_part(sweeps_bp)
        sweep_only = sweeps_bp[sweepregion]
        sweeps_fft = 20*np.log10(np.abs(np.fft.rfft(sweep_only)))
        sweeps_freq = np.fft.rfftfreq(sweep_only.size,1.0/fs)
        sweep_ffts[i] = np.column_stack((sweeps_freq, sweeps_fft))
    # make 'average' sweep spectrum
    avgd_fft, freqs = make_avged_fft('1', {'1':sweep_ffts})
    return freqs, avgd_fft



def get_avg_sweep_spectrum(fname):
    audio, fs = sf.read(fname)
    freqs, spectrum = calculate_average_sweep_spectrum(audio, fs)
    return freqs, spectrum

