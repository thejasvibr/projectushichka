'''
This module is a copy of the one used in the 2018-07-16 microphone calibration folder. 
There are a few changes made to increase readability and other such updates, though the structure is 
the same. 

Primary changes: 
* Sweep duration now 3ms from 10ms
* Silence between playbacks now 100ms

- Thejasvi Beleyur, code released with MIT license
'''
import queue
import datetime as dt
import numpy as np 
import os
import scipy.signal as signal 
import scipy.io.wavfile as WAV
#import matplotlib.pyplot as plt
#plt.rcParams['agg.path.chunksize'] = 100000
import sounddevice as sd 
import time


def make_playback_sounds():
    fs = 192000

    # define a common pl.ayback sounds length:
    common_length = 0.2
    numsamples_comlength = int(fs*common_length)

    # Create the calibration chirp : 
    chirp_durn = 0.003
    t_chirp = np.linspace(0,chirp_durn, int(fs*chirp_durn))
    start_f, end_f = 15000, 95000
    sweep = signal.chirp(t_chirp,start_f, t_chirp[-1], end_f, 'linear')
    sweep *= signal.tukey(sweep.size, 0.9)
    sweep *= 0.5

    silence_durn = 0.1
    silence_samples = int(silence_durn*fs)
    silence = np.zeros(silence_samples)
    # create 5 sweeps with silences before & after them :
    sweep_w_leftsilence = np.concatenate((silence,sweep))
    numsamples_to_add = numsamples_comlength - sweep_w_leftsilence.size
    sweep_w_silences = np.concatenate((sweep_w_leftsilence,np.zeros(numsamples_to_add)))
    sweep_w_silences = np.float32(sweep_w_silences)

    all_sweeps = []
    #make a set of 5 sweeps in a row =
    for i in range(5):
        all_sweeps.append(sweep_w_silences)

    # create a set of sinusoidal pulses :
    pulse_durn = 0.1
    pulse_samples = int(fs*pulse_durn)
    t_pulse = np.linspace(0,pulse_durn, pulse_samples)
    pulse_start_f, pulse_end_f  = 10000, 95000
    frequency_step = 1000;
    freqs = np.arange(pulse_start_f, pulse_end_f, frequency_step)

    all_freq_pulses = []

    for each_freq in freqs:
        one_tone = np.sin(2*np.pi*t_pulse*each_freq)
        one_tone *= signal.tukey(one_tone.size, 0.85)
        one_tone *= 0.5
        one_tone_w_silences = np.float32(np.concatenate((silence,one_tone)))
        all_freq_pulses.append(one_tone_w_silences)

    # setup the speaker playbacks to first play the sweeps and then 
    # the pulses : 
    playback_sounds = [all_sweeps, all_freq_pulses]
    return playback_sounds, numsamples_comlength 

def generate_todays_pbk_folder(**kwargs):
    save_path = kwargs.get('save_path','../')
    today = dt.datetime.today().strftime('%Y-%m-%d')
    intended_path = os.path.join(save_path, today)
    if not os.path.isdir(intended_path):
        os.mkdir(intended_path)
    return intended_path
    
    
    


def perform_playback(playback_sounds, mic_rec_name, fs=192000, dev_ind=40, **kwargs):
    '''
    '''
    num_in_out = kwargs.get('num_in_out',[9,1])
    data_in_channel = kwargs.get('data_in_channel',[8])
    numsamples_comlength = kwargs.get('numsamples_comlength', int(0.2*fs))
    
    q = queue.Queue()
    S = sd.Stream(samplerate=fs, blocksize=numsamples_comlength, 
                  channels=num_in_out,device=dev_ind)
    S.start()
    begin_pbk = True
    while begin_pbk:
        for each_pbk_series in  playback_sounds:
            for each_sound in each_pbk_series:		
                input_data, error_msg = S.read(numsamples_comlength)
                S.write(each_sound)
                q.put(input_data[:,data_in_channel])

        begin_pbk = False
    S.stop()
    del S
    y = [ q.get() for each_segment in range(q.qsize())]
    rec = np.concatenate(y)
    output_folder = generate_todays_pbk_folder(**kwargs)
    final_output_file = os.path.join(output_folder, mic_rec_name)
    WAV.write(final_output_file, fs, rec)

if __name__ == '__main__':

    playback_sounds, numsamples_comlength = make_playback_sounds()
    mic_num = 'gras'
    angle = '0'
    gain = '46'
    orientation='azimuth'
    kwargs = {'save_path':'../'}
    #all_pbk_sounds = np.concatenate(pbk_sounds)
    fs = 192000
    dev_ind = 40
    for i in range(2):
        timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        mic_rec_name = mic_num+'_'+'gaindB_'+gain+'_'+orientation+'_angle_'+angle+'_'+timestamp+'.wav'
        print('playback starting....')
        
        num_in_out = kwargs.get('num_in_out',[9,1])
        data_in_channel = kwargs.get('data_in_channel',[8])
        numsamples_comlength = kwargs.get('numsamples_comlength', int(0.2*fs))
        
        q = queue.Queue()
        S = sd.Stream(samplerate=fs, blocksize=numsamples_comlength, 
                      channels=num_in_out,device=dev_ind)
        S.start()
        begin_pbk = True
        while begin_pbk:
            for each_pbk_series in  playback_sounds:
                for each_sound in each_pbk_series:		
                    input_data, error_msg = S.read(numsamples_comlength)
                    S.write(each_sound)
                    q.put(input_data[:,data_in_channel])

            begin_pbk = False
        S.stop()
        del S
        y = [ q.get() for each_segment in range(q.qsize())]
        rec = np.concatenate(y)
        output_folder = generate_todays_pbk_folder(**kwargs)
        final_output_file = os.path.join(output_folder, mic_rec_name)
        WAV.write(final_output_file, fs, rec)
        print('playback done....sleeping a bit...')
        time.sleep(5)
        
