'''
This module is a copy of the one used in the 2018-07-16 microphone calibration folder. 
There are a few changes made to increase readability and other such updates, though the structure is 
the same. 
'''
import Queue
import datetime as dt
import numpy as np 
import scipy.signal as signal 
import scipy.io.wavfile as WAV
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 100000
import sounddevice as sd 

mic_num = 'SMP1'
gain = '30'
orientation='azimuth'
angle = '0'
timenow = dt.datetime.now()
timestamp = timenow.strftime('%Y-%m-%d_%H-%M-%S')
mic_rec_name = mic_num+'_'+'gaindB_'+gain+'_'+orientation+'_angle_'+angle+'_'+timestamp+'.wav'

fs = 192000

# define a common pl.ayback sounds length:
common_length = 0.7
numsamples_comlength = int(fs*common_length)

# Create the calibration chirp : 
chirp_durn = 0.010
t_chirp = np.linspace(0,chirp_durn, int(fs*chirp_durn))
start_f, end_f = 15000, 95000
sweep = signal.chirp(t_chirp,start_f, t_chirp[-1], end_f, 'linear')
sweep *= signal.tukey(sweep.size, 0.9)
sweep *= 0.5

silence_durn = 0.3
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
    one_tone_w_silences = np.float32(np.concatenate((silence,one_tone,silence)))
    all_freq_pulses.append(one_tone_w_silences)

# setup the speaker playbacks to first play the sweeps and then 
# the pulses : 
playback_sounds = [all_sweeps, all_freq_pulses]

q = Queue.Queue()

dev_ind = 56
S = sd.Stream(samplerate=fs, blocksize=numsamples_comlength, 
              channels=[1,1],device=dev_ind)
S.start()
begin_pbk = True
while begin_pbk:
    for each_pbk_series in  playback_sounds:
        for each_sound in each_pbk_series:		
            input_data, error_msg = S.read(numsamples_comlength)
            S.write(each_sound)
            q.put(input_data)

	begin_pbk = False
S.stop()
y = [ q.get() for each_segment in range(q.qsize())]
rec = np.concatenate(y)
output_folder = 'C:\\Users\\tbeleyur\\Documents\\fieldwork_2018\\mic_calibrations\\2018-07-16\\'
WAV.write(output_folder+mic_rec_name, fs, rec)
