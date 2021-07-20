# -*- coding: utf-8 -*-
"""
Creating mic2mic matrix for 2018-08-14
======================================
Session with 20 channels  recording !!

The order of microphone ids :
    * SMP1,2,3,4,5,6,7, SANKEN9,10,11,12, SMP8, SMP1',2',3',4',5',6',7',8'
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
good_mics = [0,1,2,3,4,5,6,
             8,9,10,11,
             14,15,16,17,18,19,20,21,22]

with open('2018-07-28_goodmics.txt','w') as txtfile:
    txtfile.write(str(good_mics)[1:-1])

#%% 
# Initiate the distance matrix with just channels 0-3 in it 
R = 1.2
theta = np.pi/3
mic00 = [0,0,0]
mic01 = [-R*np.sin(theta),  0, -R*np.cos(theta)]
mic02 = [R*np.sin(theta),  0, -R*np.cos(theta)]# make it large to then replace with np.nan
mic03 = [0, 0, R]

tristar = np.row_stack((mic00, mic01, mic02, mic03))
tristar_distmat = spatial.distance_matrix(tristar, tristar)

#%% This refers to the MICROPHONE identitites, and not the channel IDs
# These measurements are taken from the field notes.
smp_mic2mic = {}
smp_mic2mic['1,8p'] = 4.098
smp_mic2mic['1,7p'] = 3.300
smp_mic2mic['1,6p'] = np.round(np.mean([3.314, 3.27]), 3)
smp_mic2mic['2,8p'] = np.round(np.mean([4.126, 4.132]), 3)
smp_mic2mic['2,7p'] = np.round(np.mean([3.239, 3.236]), 3)
smp_mic2mic['2,6p'] = np.round(np.mean([3.336, 3.317]), 3)
smp_mic2mic['4,8p'] = 3.323
smp_mic2mic['4,7p'] = np.round(np.mean([2.335, 2.340]), 3)
smp_mic2mic['4,6p'] = np.round(np.mean([2.204, 2.185]), 3)
smp_mic2mic['5,5p'] = np.round(np.mean([3.381, 3.378]), 3)
smp_mic2mic['5,7p'] = np.round(np.mean([2.272, 2.265]), 3)
smp_mic2mic['5,6p'] = np.round(np.mean([2.246, 2.266]), 3)
smp_mic2mic['6,8p'] = 3.481
smp_mic2mic['6,7p'] = np.round(np.mean([2.252, 2.249]), 3)
smp_mic2mic['6,6p'] = np.round(np.mean([2.426, 2.384]), 3)
smp_mic2mic['7,8p'] = np.round(np.mean([2.892, 2.893]), 3)
smp_mic2mic['7,7p'] = np.round(np.mean([1.708, 1.666]), 3)
smp_mic2mic['7,7p'] = np.round(np.mean([1.708, 1.666]), 3)
smp_mic2mic['8,8p'] = np.round(np.mean([2.896, 2.906]), 3)
smp_mic2mic['8,7p'] = np.round(np.mean([1.507, 1.510]), 3)
smp_mic2mic['1p,8p'] = 2.962
smp_mic2mic['1p,7p'] = np.round(np.mean([1.432, 1.420]), 3)
smp_mic2mic['1p,6p'] = np.round(np.mean([1.503, 1.457]), 3)
smp_mic2mic['2p,8p'] = np.round(np.mean([3.715, 3.733]), 3)
smp_mic2mic['2p,7p'] = np.round(np.mean([2.752, 2.750]), 3)
smp_mic2mic['2p,6p'] = np.round(np.mean([2.774, 2.737]), 3)
smp_mic2mic['3p,8p'] = np.round(np.mean([3.749, 3.764]), 3)
smp_mic2mic['3p,7p'] = 2.710
smp_mic2mic['4p,8p'] = 3.03
smp_mic2mic['3p,6p'] = np.round(np.mean([2.848, 2.826]), 3)
smp_mic2mic['4p,7p'] = np.round(np.mean([1.827, 1.838]), 3)
smp_mic2mic['4p,6p'] = np.round(np.mean([1.655, 1.605]), 3)
smp_mic2mic['5p,8p'] = np.round(np.mean([3.074, 3.079]), 3)
smp_mic2mic['5p,7p'] = np.round(np.mean([1.711, 1.715]), 3)
smp_mic2mic['5p,6p'] = np.round(np.mean([1.719, 1.733]), 3)

# #%% Make the mapping between SMP mic number and the final channel number (0-indexed)
# smp_to_channel = {1:4, 2:5, 3: None, 4: 6, 5:7, 6:8, 7:9, 8:10}

# #%% 

# mic2mic_distmat = np.empty((len(good_mics), len(good_mics)))
# mic2mic_distmat[:] = np.nan
# mic2mic_distmat[:4,:4] = tristar_distmat

# for pair, distance in smp_mic2mic.items():
#     smp_mica, smp_micb = pair

#     channel_mica = smp_to_channel[smp_mica]
#     channel_micb = smp_to_channel[smp_micb]
#     if not np.logical_or(channel_mica is None, channel_micb is None):
#         mic2mic_distmat[channel_mica, channel_micb] = distance

# #%% Some more SMP-SANKEN 9 distances (SANKEN9 is the first channel):
# smp_to_sanken9  = {}
# smp_to_sanken9[1] = np.round(np.mean([3.618, 3.612]), 3)
# smp_to_sanken9[2] = np.round(np.mean([3.471, 3.455]), 3)
# smp_to_sanken9[4] = np.round(np.mean([3.204, 3.198]), 3)
# smp_to_sanken9[5] = np.round(np.mean([3.371, 3.316]), 3)
# smp_to_sanken9[6] = np.round(np.mean([2.955, 2.965]), 3)
# smp_to_sanken9[8] = np.round(np.mean([2.026, 2.028]), 3)

# for smp_mic, distance in smp_to_sanken9.items():
#     channel_mic = smp_to_channel[smp_mic]
#     mic2mic_distmat[channel_mic,0] = distance

# #%% make the matrix symmetric across the diagonal the 'tedious' way:

# upper_or_lower = {0:'lower',1:'upper'}
# for i in range(mic2mic_distmat.shape[0]):
#     for j in range(mic2mic_distmat.shape[0]):
#         lower = mic2mic_distmat[i,j]
#         upper = mic2mic_distmat[j,i]
        
#         are_nans = [np.isnan(lower), np.isnan(upper)]

#         if sum(are_nans)==0:
#             pass
#         elif sum(are_nans)==1:
#             non_nan_value = int(np.argwhere(np.invert(are_nans)))
#             if upper_or_lower[non_nan_value]=='lower':
#                 mic2mic_distmat[j,i] = mic2mic_distmat[i,j]
#             elif upper_or_lower[non_nan_value]=='upper':
#                 mic2mic_distmat[i,j] = mic2mic_distmat[j,i]
#         elif sum(are_nans) ==2:
#             pass

# #%% 
# distmat = pd.DataFrame(mic2mic_distmat)
# distmat.columns = [ 'channel'+str(each+1) for each in range(len(good_mics))]
# distmat.index = [ 'channel'+str(each+1) for each in range(len(good_mics))]

# #%% Write the distance matrix:
# distmat.to_csv('mic2mic_RAW_2018-07-28.csv',na_rep='NaN')