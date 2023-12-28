# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:00:29 2023

@author: theja
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from nptdms import TdmsFile
from skimage.transform import resize
import scipy.signal as signal 
import os
from datetime import datetime as dt
import tqdm

tdms_files = glob.glob('2023-12-18-Nuernbergzoo_arjanboonman/*.tdms')

#%%
# 2023-12-18_17-55-25 is good (2-3 bats a time)
# single bat: 17-47-03
#
current_file = tdms_files[3]
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


#%%
# Try to align the acoustic cam audio and the known 500kHz audio data
all_audio_files = glob.glob('ch3\*.WAV')
creation_times = [dt.fromtimestamp(os.path.getctime(each)) for each in all_audio_files]
fs, raw_avisoft = io.wavfile.read(all_audio_files[3])
raw_avisoft = np.float64(raw_avisoft)
raw_avisoft /= 2**15

contraction_factor = 48/500
downsampled_audio  = signal.resample(raw_avisoft, int(raw_avisoft.size*contraction_factor))


fs = int(48e3)
b,a = signal.butter(3, np.array([1e-1,1.5e3])/(2*fs), 'bandpass')
avisoft_bp = signal.filtfilt(b, a, downsampled_audio)
avisoft_bp /= abs(avisoft_bp).max()
io.wavfile.write('avisoft_downsample.wav', rate=int(48e3), data=downsampled_audio)

bp_accam = signal.filtfilt(b, a, full_audio)
bp_accam /= abs(bp_accam).max()
#%%
full_cc = signal.correlate(avisoft_bp, bp_accam[-3*fs:], 'full')

plt.plot(full_cc)
plt.vlines(full_cc.size*0.5, 0, full_cc.max(),'r')
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
import matplotlib.animation as animation

filtered_beams_images = beams_images.copy()
filtered_beams_images -= np.percentile(beams_images, q=2, axis=0)

filtered_cam = image_interpolated.copy()
filtered_cam -= np.percentile(image_interpolated, q=5,  axis=0)
filtered_cam += abs(filtered_cam.min()) 
filtered_cam = filtered_cam**2


#%%
from matplotlib import gridspec

mixed_datastreams = image_interpolated + beams_images*15
vidmixfig = plt.figure()
gs = gridspec.GridSpec(4, 2) 
spectrumax = plt.subplot(gs[0,:])
vidmixax = plt.subplot(gs[1:,:])




mix_artist = vidmixax.imshow(mixed_datastreams[0,:,:], vmin=mixed_datastreams.min(), 
                             vmax=mixed_datastreams.max())
framenum_artist = vidmixax.set_title(f'Frame number: {0}')

spectrumax.specgram(full_audio, Fs=int(48e3), NFFT=512, noverlap=256)
now_line = spectrumax.plot([0,0], [0,20e3], 'k-')[0]
vidmixax.set_yticks([]); vidmixax.set_xticks([]);

def update_vid(framenum):
    #print(f'Framenum saving: {framenum}')
    framenum_artist.set_text(f'Frame number: {framenum}')
    mix_artist.set_data(mixed_datastreams[framenum,:,:])
    now_line.set_xdata(framenum/100)
    return mix_artist


def progress_func(currframe, total_frames):
    print(f'{np.around(currframe/total_frames,2)} complete')
    

video_filename = f'{tdms_path_name}_audiovideo.mp4'
vid_ani = animation.FuncAnimation(fig=vidmixfig, func=update_vid,
                                  frames=range(image_interpolated.shape[0]), 
                              interval=10).save(video_filename, progress_callback=progress_func,fps=100, dpi=200)
#plt.show()




#%%

vidfig, vidax = plt.subplots(1,1)
imshow_artist = vidax.imshow(image_interpolated[0,:,:], vmin=image_interpolated.min(), 
                             vmax=image_interpolated.max())
vidax.set_yticks([]); vidax.set_xticks([]);

def update_vid(framenum):
    #print(framenum)
    imshow_artist.set_data(image_interpolated[framenum,:,:])
    return imshow_artist

video_filename = f'{tdms_path_name}_onlyvideo.mp4'
vid_ani = animation.FuncAnimation(fig=vidfig, func=update_vid,
                                  frames=range(image_interpolated.shape[0]),
                              interval=10).save(video_filename, fps=100)
#plt.show()
