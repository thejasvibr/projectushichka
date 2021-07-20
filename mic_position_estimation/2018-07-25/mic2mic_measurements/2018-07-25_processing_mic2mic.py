# -*- coding: utf-8 -*-
"""
Creating mic2mic matrix for 2018-07-25
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

with open('2018-07-25_goodmics.txt','w') as txtfile:
    txtfile.write(str(good_mics)[1:-1])

#%% 
# Initiate the distance matrix with just channels 0-3 in it 
R = 1.2
theta = np.pi/3
mic00 = [0,0,0]
mic01 = [999, 999,999] # make it large to then replace with np.nan
mic02 = [R*np.sin(theta),  0, -R*np.cos(theta)]
mic03 = [0, 0, R]

tristar = np.row_stack((mic00, mic01, mic02, mic03))
tristar_distmat = spatial.distance_matrix(tristar, tristar)
tristar_distmat[tristar_distmat>3] = np.nan

#%% This refers to the MICROPHONE identitites, and not the channel IDs
# These measurements are taken from the field notes.
smp_mic2mic = {}
smp_mic2mic[(1,2)] = 0.691
smp_mic2mic[(1,3)] = np.round(np.mean([1.257, 1.235]),3)
smp_mic2mic[(1,4)] = np.round(np.mean([1.827, 1.840]), 3)
smp_mic2mic[(1,5)] = np.round(np.mean([2.300, 2.22]), 3)
smp_mic2mic[(1,6)] = np.round(np.mean([2.637, 2.635]), 3)
smp_mic2mic[(1,7)] = np.round(np.mean([3.649, 3.659]), 3)
smp_mic2mic[(1,8)] = np.round(np.mean([3.694, 3.687]), 3)
smp_mic2mic[(2,4)] = 1.490
smp_mic2mic[(2,6)] = np.round(np.mean([2.479, 2.485]), 3)
smp_mic2mic[(2,7)] = np.round(np.mean([3.602, 3.612]), 3)
smp_mic2mic[(2,8)] = np.round(np.mean([3.567, 3.567]), 3)
smp_mic2mic[(3,4)] = np.round(np.mean([0.830, 0.832]), 3)
smp_mic2mic[(3,5)] = np.round(np.mean([1.176, 1.187]), 3)
smp_mic2mic[(3,6)] = np.round(np.mean([1.533, 1.537]), 3)
smp_mic2mic[(3,7)] = np.round(np.mean([2.825, 2.816]), 3)
smp_mic2mic[(3,8)] = np.round(np.mean([2.903, 2.911]), 3)
smp_mic2mic[(4,5)] = np.round(np.mean([0.421, 0.420]), 3)
smp_mic2mic[(4,6)] = 1.068
smp_mic2mic[(4,7)] = 2.289
smp_mic2mic[(4,8)] = 2.257
smp_mic2mic[(5,6)] = 0.716
smp_mic2mic[(5,7)] = 1.919
smp_mic2mic[(5,8)] = 1.904
smp_mic2mic[(7,8)] = 0.613

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
smp_to_sanken9[1] = np.round(np.mean([4.768, 4.773]), 3)
smp_to_sanken9[2] = np.round(np.mean([4.774, 4.774]), 3)
smp_to_sanken9[4] = 4.290
smp_to_sanken9[5] = 4.200
smp_to_sanken9[6] = 4.343
smp_to_sanken9[7] = 3.466
smp_to_sanken9[8] = 3.099

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
distmat.to_csv('mic2mic_RAW_2018-07-25.csv',na_rep='NaN')    
