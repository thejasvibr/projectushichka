# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:00:29 2023

Avisoft-acoustic camera correspondence list
-------------------------------------------

|Acoustic camera timestamp | Avisoft recording number |
|--------------------------|--------------------------|
| 2023-12-18_18-12-48      | ch3\\T0000036            |
| 2023-12-18_18-11-54      | ch3_T0000035             |
| 2023-12-18_18-11-34      | ch3\\T0000034            |
| 2023-12-18_18-10-37      | ch2\\T0000033            |
| 2023-12-18_18-10-06  | ch3\\T0000032.wav 
| 2023-12-18_18-09-29  | ch3\T0000031.wav
| 2023-12-18_18-09-02  | ch3\\T0000030.wav
| 2023-12-18_18-01-55  |  ch3\\T0000028.wav
| 2023-12-18_17-58-14  | ch3\\T0000026.wav
| 2023-12-18_17-57-36  | ch3\\T0000025.wav
| 2023-12-18_17-57-19 | ch3\\T0000024.wav
| 2023-12-18_17-56-58 | ch3\\T0000023.wav
| 2023-12-18_17-55-48 | ch3\\T0000022.wav |
| 2023-12-18_17-55-25|ch3\\T0000019.wav| ~ 7 sec in the video - tehre's upto 3 spots
@author: theja
"""

import glob
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import scipy.io as io
from nptdms import TdmsFile
from skimage.transform import resize
import sounddevice as sd
import scipy.signal as signal 
import os
from datetime import datetime as dt
import tqdm

to_dB = lambda X: 20*np.log10(X)

tdms_files = glob.glob('2023-12-18-Nuernbergzoo_arjanboonman/*.tdms')

#%%
# 2023-12-18_17-55-25 is good (2-3 bats a time)
# single bat: 17-47-03
#
current_file = tdms_files[16]
tdms_path_name = os.path.split(current_file)[-1][:-5]

tdms_file = TdmsFile.read(current_file)
beams = tdms_file['beams']

beams_rows, beams_cols = 48, 64


frames = tdms_file['pictures']
frames_rows, frames_cols = 240, 320

# Add the negative and postive sample data to get the full audio data (of the
# central mic?)
neg_audio = tdms_file['timedata'].channels()[0].data
pos_audio = tdms_file['timedata'].channels()[1].data
full_audio = pos_audio+neg_audio
full_audio = full_audio.astype(np.float64)

full_audio /= np.max(np.abs(full_audio))
full_audio *= 0.9
io.wavfile.write(filename=f'{tdms_path_name}_48kHz_audio.wav', rate=int(48e3), data=full_audio)
print(tdms_path_name)

#%%
# Try to align the acoustic cam audio and the known 500kHz audio data
all_audio_files = glob.glob('ch3\*.WAV')
creation_times = [dt.fromtimestamp(os.path.getctime(each)) for each in all_audio_files]
audio_file_path = all_audio_files[16]
audio_file_savename = audio_file_path.replace('\\','_')[:-4]
fs_avisoft, raw_avisoft = io.wavfile.read(audio_file_path)
print(audio_file_savename)

raw_avisoft = np.float64(raw_avisoft)
raw_avisoft /= 2**15

contraction_factor = 48/500
downsampled_audio  = signal.resample(raw_avisoft, int(raw_avisoft.size*contraction_factor))

# perform the cross correlation based on the voices -  this is the best option
# after a bit of trial and error. Therefore we filter out all the higher frequencise
# to avoid the frogs and the aliased bat calls.
fs = int(48e3)
b,a = signal.butter(2, np.array([100, 1.5e3])/(2*fs), 'bandpass')
avisoft_bp = signal.lfilter(b, a, downsampled_audio)
avisoft_bp /= abs(avisoft_bp).max()
io.wavfile.write(f'avisoft_downsample_{audio_file_savename}.wav', rate=int(48e3), data=downsampled_audio)

bp_accam = signal.lfilter(b, a, full_audio)
bp_accam /= abs(bp_accam).max()
#%%
# exploit the fact that the acoustic camera has a fixed 10 sec pre-buffer saving
# and that Arjan always said 'yes'/'okay'when he was about to press save.
accam_lastfew = bp_accam[int(-5*fs):]

cc = signal.correlate(avisoft_bp, accam_lastfew,  'same')
plt.figure()
plt.subplot(411)
plt.plot(np.arange(cc.size)/fs,cc)
maxind = np.argmax(cc)
plt.plot(maxind/fs, cc[maxind],'k*')
a2 = plt.subplot(412)
#plt.plot(accam_lastfew, label='accam')
plt.specgram(accam_lastfew, Fs=fs, NFFT=512, noverlap=256)
a3 = plt.subplot(413, sharex=a2)
#plt.plot(avisoft_bp, label='avisoft')
plt.specgram(avisoft_bp, Fs=fs, NFFT=512, noverlap=256)
plt.subplot(414, sharex=a3)
start_ind = maxind - int(accam_lastfew.size*0.5)
accam_lastfew_endind = start_ind+accam_lastfew.size
plt.specgram(accam_lastfew, Fs=fs, NFFT=512, noverlap=256,
             xextent=[start_ind/fs, accam_lastfew_endind/fs])

timestart_relavisoft = (accam_lastfew_endind - full_audio.size)/fs
avisoft_full_start = -timestart_relavisoft

#t_startavisoft 
#%%
plt.figure()
a1 = plt.subplot(411)
plt.specgram(full_audio, Fs=fs,  NFFT=512, noverlap=256)
plt.ylim(0,8e3)
plt.subplot(412, sharex=a1)
plt.plot(np.linspace(0,full_audio.size/fs,full_audio.size), full_audio)


a2 = plt.subplot(413, sharex=a1)
plt.specgram(downsampled_audio, Fs=fs, NFFT=512, noverlap=256,
             xextent=[avisoft_full_start, avisoft_full_start+ downsampled_audio.size/fs])
plt.ylim(0,8e3)

plt.subplot(414, sharex=a1)
plt.plot(np.linspace(0,downsampled_audio.size/fs,downsampled_audio.size)+avisoft_full_start,
                                                 downsampled_audio)
#%%
# Make a stereo wav file with the aligned audio tracks to check again. 
tstartstop_both = np.array([[0, full_audio.size/fs],
                             [avisoft_full_start,
                                     avisoft_full_start+downsampled_audio.size/fs]])

tstartstop_both -= tstartstop_both.min()
final_size = int(tstartstop_both.max()*fs) + 100
stereo_track = np.zeros((final_size,2))

tstastop_indices = np.int64(tstartstop_both*fs)
tstastop_indices[0,1] = tstastop_indices[0,0] + full_audio.size
tstastop_indices[1,1] = tstastop_indices[1,0] + downsampled_audio.size
# Track 1 is the acoustic camera central microphone
stereo_track[tstastop_indices[0,0]:tstastop_indices[0,1],0] += full_audio
stereo_track[tstastop_indices[1,0]:tstastop_indices[1,1],1] += downsampled_audio
io.wavfile.write(f'{tdms_path_name}_aligned_stereo_track.wav', fs, stereo_track)

#%%
for target  in ['camera_frames', 'beam_images']:
    if not os.path.exists(target):
        os.mkdir(target)
    if not os.path.exists(os.path.join(target, tdms_path_name)):
        os.mkdir(os.path.join(target, tdms_path_name))

beams_images = []
for i, every in tqdm.tqdm(enumerate(beams.channels())):
    try:
        resized_image = resize(every.data.reshape(beams_rows,beams_cols),
                               (frames_rows,frames_cols))
        beams_images.append(resized_image)
    except:
        pass
beams_images = np.array(beams_images)


frames_data = []
for i, frame in tqdm.tqdm(enumerate(frames.channels())):
    try:
        frames_data.append(frame.data.reshape(frames_rows, frames_cols))
    except:
        pass

frames_data = np.array(frames_data)

#%% Now upsample the image data to the frame rate of the 
from itertools import product
from scipy.interpolate import interp1d

ij_all = product(range(frames_rows), range(frames_cols))
t_camera = frames.channels()[-1].data
t_beams = beams.channels()[-1].data
image_interpolated = np.zeros((beams_images.shape))

for (i,j)  in tqdm.tqdm(ij_all):
    interp_fun = interp1d(t_camera, frames_data[:,i,j], fill_value='extrapolate')
    image_interpolated[:,i,j] = interp_fun(t_beams)

#%%


filtered_beams_images = beams_images.copy()
filtered_beams_images -= np.percentile(beams_images, q=2, axis=0)

filtered_cam = image_interpolated.copy()
filtered_cam -= np.percentile(image_interpolated, q=5,  axis=0)
filtered_cam += abs(filtered_cam.min()) 
filtered_cam = filtered_cam**2


#%%

def clip_image_dynamicrange(image, dynamic_range=80, good_dyn_range=3):
    '''Converts image to dB scale, normalises
    to highest pixel value and clips to the set 
    dynamic range. 
    '''
    dB_image = to_dB(image)
    dB_image -= np.max(dB_image)
    if np.max(dB_image)-np.min(dB_image)<=good_dyn_range:
        dB_image[:,:] = -dynamic_range
    else:
        dB_image[dB_image<-dynamic_range] = -dynamic_range

    return dB_image
#%%
b,a = signal.butter(2, np.array([20e3, 180e3])/(500e3*2), 'bandpass')
bp_avisoft = signal.filtfilt(b,a, raw_avisoft)
bp_avisoft /= np.max(np.abs(bp_avisoft))

from matplotlib import gridspec

mixed_datastreams = image_interpolated + beams_images*20

vidmixfig = plt.figure()
gs = gridspec.GridSpec(25, 2) 
spectrum500 = plt.subplot(gs[:4,:])
spectrumax = plt.subplot(gs[6:10,:])

vidmixax = plt.subplot(gs[12:,:])
 


mix_artist = vidmixax.imshow(mixed_datastreams[0,:,:], vmin=mixed_datastreams.min(), 
                             vmax=mixed_datastreams.max())
vidmixax.set_yticks([]); vidmixax.set_xticks([]);

framenum_artist = vidmixax.set_xlabel(f'Frame number: {0}')
fullaudio_kwargs = {'fs':fs, 'nperseg':128, 'noverlap':64,}
f,t,sxx = signal.spectrogram(full_audio[:int(0.2*fs)],
                              **fullaudio_kwargs)

fullspec_artist = spectrumax.imshow(clip_image_dynamicrange(sxx),
                                    aspect='auto', origin='lower',
                                    vmax=0,
                                    vmin=-80)
spectrumax.set_yticks([0, sxx.shape[0]])
spectrumax.set_yticklabels([str(0), str(f[-1]*1e-3)])

spectrumax.set_xticks([0,  sxx.shape[1]],
                      [0, np.around(t[-1],3)])
spectrumax.set_ylabel('Ac. camera')

audio500_kwargs = {'fs':fs_avisoft, 'nperseg':128, 'noverlap':64}
f500,t500,sxx500 = signal.spectrogram(bp_avisoft[:int(0.2*fs_avisoft)],
                              **audio500_kwargs)
f500spec_artist = spectrum500.imshow(clip_image_dynamicrange(sxx500),
                                     aspect='auto', origin='lower', vmax=0,
                                     vmin=-80)
relevant_rows = np.where(np.logical_and(f500>=10e3, f500<=150e3))
spectrum500.set_ylabel('Condenser mic')

spectrum500.set_xticks([])
spectrum500.set_ylim(np.min(relevant_rows), np.max(relevant_rows))
fmin = np.around(f500[relevant_rows[0][0]]*1e-3, 2)
fmax = np.around(f500[relevant_rows[0][-1]]*1e-3, 2)
spectrum500.set_yticks([0, np.max(relevant_rows)], [fmin, fmax])


now_line = spectrumax.plot(np.tile(sxx.shape[1]*.5,2), [0,sxx.shape[0]], 'w-')[0]
now_line2 = spectrum500.plot(np.tile(sxx500.shape[1]*.5,2), [0, sxx500.shape[0]], 'w-')[0]

def update_vid(framenum):
    #print(f'Framenum saving: {framenum}')
    framenum_artist.set_text(f'Time: {framenum/100}, Frame number: {framenum}')
    mix_artist.set_data(mixed_datastreams[framenum,:,:])
    
    # update spectrograms every 200 ms
    
    t_start = framenum*1e-2 - 0.1
    if t_start < 0:
        t_start = framenum*1e-2
    t_stop = t_start + 0.2
    start_ind = int(t_start*fs)
    stop_ind = int(t_stop*fs)
    f,t,sxx = signal.spectrogram(full_audio[start_ind:stop_ind],
                                  **fullaudio_kwargs)
    fullspec_artist.set_array(clip_image_dynamicrange(sxx))
    t_midpoint = np.around((np.around(t_stop,3)-np.around(t_start,3))/2 + t_start, 2)
    spectrumax.set_xticks([0, sxx.shape[1]*0.5, sxx.shape[1]],  
                          [np.around(t_start,3), t_midpoint, np.around(t_stop,3)])
    
    
    
    t_avisoft_start = t_start-avisoft_full_start
    if t_avisoft_start >= 0:
        t_avisoft_stop = t_avisoft_start + 0.2
        ind500_start = int(t_avisoft_start*fs_avisoft)
        ind500_stop = int(t_avisoft_stop*fs_avisoft)
        f500,t500,sxx500 = signal.spectrogram(bp_avisoft[ind500_start:ind500_stop],
                                      **audio500_kwargs)
        f500spec_artist.set_array(clip_image_dynamicrange(sxx500))
        spectrum500.set_title('')
        spectrum500.set_xticks([])
    else:
        f500spec_artist.set_array(np.ones((250,500))*-80)
        spectrum500.set_title('NO HIGH-SAMPLERATE AUDIO')
        
        spectrum500.set_xticks([250],
                              ['No high sampling rate audio'])


    return mix_artist, fullspec_artist, f500spec_artist


def progress_func(currframe, total_frames):
    if np.remainder(currframe, 20)==0:
        print(f'{np.around(currframe/total_frames,2)} complete')
    

video_filename = f'{tdms_path_name}_audiovideo.mp4'
vid_ani = animation.FuncAnimation(fig=vidmixfig, func=update_vid,
                                  frames=range(0, image_interpolated.shape[0]), 
                              interval=10).save(video_filename,
                                                progress_callback=progress_func,
                                                fps=10, dpi=200)





#%%

# vidfig, vidax = plt.subplots(1,1)
# imshow_artist = vidax.imshow(image_interpolated[0,:,:], vmin=image_interpolated.min(), 
#                              vmax=image_interpolated.max())
# vidax.set_yticks([]); vidax.set_xticks([]);

# def update_vid(framenum):
#     #print(framenum)
#     imshow_artist.set_data(image_interpolated[framenum,:,:])
#     return imshow_artist

# video_filename = f'{tdms_path_name}_onlyvideo.mp4'
# vid_ani = animation.FuncAnimation(fig=vidfig, func=update_vid,
#                                   frames=range(image_interpolated.shape[0]),
#                               interval=10).save(video_filename, fps=100)
# #plt.show()