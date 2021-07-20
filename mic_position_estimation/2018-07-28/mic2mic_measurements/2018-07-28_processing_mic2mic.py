# -*- coding: utf-8 -*-
"""
Creating mic2mic matrix for 2018-07-28
======================================


@author: tbeleyur
"""

import glob
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as spatial


#%% define the good channels and write it to a file
good_mics = [0,1,2,3,4,5, 9,10,11,12,13]

with open('2018-07-28_goodmics.txt','w') as txtfile:
    txtfile.write(str(good_mics)[1:-1])

#%% 
# Initiate the distance matrix with just channels 0-3 in it 
R = 1.2
theta = np.pi/3
mic00 = [0,0,0]
mic01 = [-R*np.sin(theta),  0, -R*np.cos(theta)]
mic02 = [999, 999,999] # make it large to then replace with np.nan
mic03 = [0, 0, R]

tristar = np.row_stack((mic00, mic01, mic02, mic03))
tristar_distmat = spatial.distance_matrix(tristar, tristar)
tristar_distmat[tristar_distmat>3] = np.nan

#%% This refers to the MICROPHONE identitites, and not the channel IDs
# These measurements are taken from the field notes.
smp_mic2mic = {}
smp_mic2mic[(1,2)] = 1.113
smp_mic2mic[(1,6)] = np.round(np.mean([3.256, 3.255]),3)
smp_mic2mic[(1,7)] = np.round(np.mean([3.499, 3.599]),3)
smp_mic2mic[(1,8)] = np.round(np.mean([4.183, 4.177]),3)
smp_mic2mic[(2,4)] = np.round(np.mean([1.186, 1.186]),3)
smp_mic2mic[(2,6)] = np.round(np.mean([3.146, 3.108]),3)
smp_mic2mic[(2,7)] = 3.057
smp_mic2mic[(2,8)] = 4.074
smp_mic2mic[(4,6)] = 2.117
smp_mic2mic[(4,7)] = np.round(np.mean([2.043, 2.045]),3)
smp_mic2mic[(4,8)] = np.round(np.mean([3.253, 3.248]),3)
smp_mic2mic[(5,6)] = np.round(np.mean([1.374, 1.373]),3)
smp_mic2mic[(5,8)] = 2.821
smp_mic2mic[(6,8)] = np.round(np.mean([1.715, 1.702]),3)


#%% Make the mapping between SMP mic number and the final channel number (0-indexed)
smp_to_channel = {1:4, 2:5, 3: None, 4: 6, 5:7, 6:8, 7:9, 8:10}

#%% 

mic2mic_distmat = np.empty((len(good_mics), len(good_mics)))
mic2mic_distmat[:] = np.nan
mic2mic_distmat[:4,:4] = tristar_distmat

for pair, distance in smp_mic2mic.items():
    smp_mica, smp_micb = pair

    channel_mica = smp_to_channel[smp_mica]
    channel_micb = smp_to_channel[smp_micb]
    if not np.logical_or(channel_mica is None, channel_micb is None):
        mic2mic_distmat[channel_mica, channel_micb] = distance

#%% Some more SMP-SANKEN 9 distances (SANKEN9 is the first channel):
smp_to_sanken9  = {}
smp_to_sanken9[1] = np.round(np.mean([3.618, 3.612]), 3)
smp_to_sanken9[2] = np.round(np.mean([3.471, 3.455]), 3)
smp_to_sanken9[4] = np.round(np.mean([3.204, 3.198]), 3)
smp_to_sanken9[5] = np.round(np.mean([3.371, 3.316]), 3)
smp_to_sanken9[6] = np.round(np.mean([2.955, 2.965]), 3)
smp_to_sanken9[8] = np.round(np.mean([2.026, 2.028]), 3)

for smp_mic, distance in smp_to_sanken9.items():
    channel_mic = smp_to_channel[smp_mic]
    mic2mic_distmat[channel_mic,0] = distance

#%% make the matrix symmetric across the diagonal the 'tedious' way:

upper_or_lower = {0:'lower',1:'upper'}
for i in range(mic2mic_distmat.shape[0]):
    for j in range(mic2mic_distmat.shape[0]):
        lower = mic2mic_distmat[i,j]
        upper = mic2mic_distmat[j,i]
        
        are_nans = [np.isnan(lower), np.isnan(upper)]

        if sum(are_nans)==0:
            pass
        elif sum(are_nans)==1:
            non_nan_value = int(np.argwhere(np.invert(are_nans)))
            if upper_or_lower[non_nan_value]=='lower':
                mic2mic_distmat[j,i] = mic2mic_distmat[i,j]
            elif upper_or_lower[non_nan_value]=='upper':
                mic2mic_distmat[i,j] = mic2mic_distmat[j,i]
        elif sum(are_nans) ==2:
            pass

#%% 
distmat = pd.DataFrame(mic2mic_distmat)
distmat.columns = [ 'channel'+str(each+1) for each in range(len(good_mics))]
distmat.index = [ 'channel'+str(each+1) for each in range(len(good_mics))]

#%% Write the distance matrix:
distmat.to_csv('mic2mic_RAW_2018-07-28.csv',na_rep='NaN')