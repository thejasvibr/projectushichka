# -*- coding: utf-8 -*-
"""
How does reverberation affect source level measurements? 
========================================================
I will make sources that point in certain directions and check the effect of 
reverberation on the simple cave mesh with pyroomacoustics. 


Created on Wed Aug  9 10:37:52 2023

@author: theja
"""
import stl
from stl import mesh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import scipy.signal as signal 
from scipy.signal import fftconvolve
from scipy.spatial import distance_matrix
import soundfile as sf
import pyroomacoustics as pra
import os
#%%
datafolder = '../thermo_lidar_alignment'

#%% The transformation matrix A on page 18 of Julian's thesis (Appendix)
# This 3x4 matrix transforms homogenous coordinates from the camera to 
# the LiDAR space
A = np.array(([-0.7533, 0.6353, -0.1699, -1.7994],
              [-0.6575, -0.7332, 0.1734, 1.7945],
              [-0.0144, 0.2424, 0.9701, 0.2003]))

B = pd.read_csv(os.path.join(datafolder,'v2_transformation.csv')).to_numpy()[:,1:] # round 2 post ICP

#%%

path_to_stl = "simaudio/data/smaller_slicedup_OC_evenmore.stl"

material = pra.Material(energy_absorption=0.8, scattering=0.5)

# with numpy-stl
the_mesh = mesh.Mesh.from_file(path_to_stl)
ntriang, nvec, npts = the_mesh.vectors.shape
size_reduc_factor = 1  # to get a realistic room size (not 3km)
# create one wall per triangle
walls = []
for w in range(ntriang):
    walls.append(
        pra.wall_factory(
            the_mesh.vectors[w].T / size_reduc_factor,
            material.energy_absorption["coeffs"],
            material.scattering["coeffs"],
        )
    )

#%% Now get the bat trajectories -  and choose one position. 
bat_traj = pd.read_csv(os.path.join(datafolder,
                                    'DLTdv8_data_p000_15000_3camxyzpts.csv'))
point_trajs = []
i = 0
for col1, col2, col3 in zip(np.arange(0,9,3), np.arange(1,9,3), np.arange(2,9,3)):
    colnames = bat_traj.columns[[col1, col2, col3]]
    new_df = bat_traj.loc[:,colnames]
    new_df.columns = ['x','y','z']
    new_df['frame'] = new_df.index
    new_df['trajId'] = str(i)
    point_trajs.append(new_df)
    i += 1

bat_traj = pd.concat(point_trajs).reset_index(drop=True)
#flightpoints = [pv.Sphere(radius=0.05, center=each) for each in bat_traj.loc[:,'x':'z'].to_numpy()]
# transofrm first to rough lidar-space
flightpoints_lidar = [ np.matmul(A, np.append(each,1)) for each in bat_traj.loc[:,'x':'z'].to_numpy()]
flightpoints_final = [ np.matmul(B, np.append(each,1))[:-1] for each in flightpoints_lidar]

battraj_aligned = bat_traj.copy()
for i, (idx, row) in enumerate(battraj_aligned.iterrows()):
    battraj_aligned.loc[idx,'x':'z'] = flightpoints_final[i]
#%%
# Load the cave aligned mic xyz - get it after running simple_vs_complex_mesh-congruency.py
micxyz = pd.read_csv('cavealigned_2018-08-17.csv').loc[:,'x':'z'].to_numpy()

#%%
# Create the room


call_durn = 0.004 # seconds
fs = 192000
t = np.linspace(0,call_durn,int(fs*call_durn))
bat_call = signal.chirp(t,90000, t[-1],30000)
bat_call *= signal.windows.tukey(bat_call.size, 0.95)

source_position = battraj_aligned.loc[5,'x':'z'].to_numpy(dtype=np.float64)

room = pra.Room(walls, fs=fs, max_order=1, 
                ray_tracing=True, air_absorption=True)
room.add_source(source_position, signal=bat_call)
room.add_microphone_array(micxyz.T)

room.image_source_model()
room.compute_rir()

#%% 
# Which mics are in the cave mesh
mics_inroom = [room.is_inside(each) for each in micxyz]

all_channels_audio = []
for i,(xyz, micinroom) in enumerate(zip(micxyz, mics_inroom)):
    if micinroom:
        all_channels_audio.append(fftconvolve(room.rir[i][0], bat_call))
    else:
        all_channels_audio.append([])

#%%
# Visualise the source in the cave + mics
import pyvista as pv

plotter = pv.Plotter()
cavemesh = pv.read(os.path.join('simaudio','data','smaller_slicedup_OC_evenmore.stl'))
plotter.add_mesh(cavemesh)

mic_spheres = [pv.Sphere(radius=0.05, center=each) for each in micxyz]
for each in mic_spheres:
    plotter.add_mesh(each, color='red')

source_sphere = pv.Sphere(radius=0.05, center=source_position)
plotter.add_mesh(source_sphere, color='green')

plotter.camera.position = (6.04, -1.02, -0.57)
plotter.camera.azimuth = -6
plotter.camera.roll = -98
plotter.camera.elevation = 0.5 #-15

plotter.camera.view_angle = 45
plotter.camera.focal_point = (0.89,-0.51,-0.25)
plotter.show()


#%%
mic_inds = np.where(mics_inroom)[0]
idx = 10
t = np.linspace(0,all_channels_audio[mic_inds[idx]].size/fs,
                all_channels_audio[mic_inds[idx]].size)
plt.figure()
a0 = plt.subplot(311)
plt.specgram(all_channels_audio[mic_inds[idx]], Fs=fs, NFFT=192, noverlap=191)    
a1 = plt.subplot(312, sharex=a0)
plt.plot(t, all_channels_audio[mic_inds[idx]])


def generate_acc(audio):
    return signal.correlate(audio, audio)

inputaudio = all_channels_audio[mic_inds[idx]]
acc_inputaudio = generate_acc(inputaudio)
t = np.linspace(-inputaudio.size*0.5, inputaudio.size*0.5, acc_inputaudio.size)/fs
a2 = plt.subplot(313)
a2.plot(t, acc_inputaudio)


#%%
# Load actual data and compare the audio 
audiofilepath = 'exptl_audio/video_synced10channel_first15sec_1529546136.WAV'
thisfs = sf.info(audiofilepath).samplerate
audio_snip, fs = sf.read(audiofilepath,
                         start=int(thisfs*1.147), stop=int(thisfs*1.177))

all_accs = {}
for chnum in range(audio_snip.shape[1]):
    all_accs[chnum] = generate_acc(audio_snip[:,chnum])

acc_peaks = {}
for chnum, acc in all_accs.items():
    acc_mid = int(acc.size*0.5)
    positive_acc = acc[acc_mid:]
    envelope = abs(signal.hilbert(positive_acc))
    peaks, heights = signal.find_peaks(envelope,
                                       height=envelope.max()*0.2,
                                       distance=int(fs*0.5e-3)
                                       )
    plt.figure()
    plt.plot(positive_acc)
    plt.plot(peaks, envelope[peaks],'r')
    plt.title(f'Channel {chnum}')


# plt.figure()
# a0 = plt.subplot(311)
# plt.specgram(audio_snip[:,focalch], Fs=fs, NFFT=192, noverlap=191)    
# a1 = plt.subplot(312, sharex=a0)

# t = np.linspace(0,audio_snip[:,focalch].size/fs,
#                 audio_snip[:,focalch].size)
# plt.plot(t, audio_snip[:,focalch])

# inputaudio = audio_snip[:,focalch]
# acc_inputaudio = generate_acc(inputaudio)
# t = np.linspace(-inputaudio.size*0.5, inputaudio.size*0.5, acc_inputaudio.size)/fs
# a2 = plt.subplot(313)
# a2.plot(t, acc_inputaudio)


# inputaudio = audio_snip[:,2]
# acc_inputaudio = generate_acc(inputaudio)
# acc_env = np.abs(signal.hilbert(acc_inputaudio))
# t = np.linspace(-inputaudio.size*0.5, inputaudio.size*0.5, acc_inputaudio.size)/fs
