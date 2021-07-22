# -*- coding: utf-8 -*-
"""
Creating mic2mic matrix for 2018-08-19
======================================
Session with 20 channels  recording !!

The order of microphone ids :
    * SMP1,2,3,4,5,6,7, SANKEN9,10,11,12, SMP8, SMP1',2',3',4',5',6',7',8'

Microphones SMP8 and SMP1' aren't included because the audio is digitised
thorugh the Focusrite OctoPre Scarlett, and the ADC delay is now known. 

Some mics not working
---------------------
SMP3 isn't working - even in the evening recordings too. 


@author: tbeleyur
"""


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import scipy.spatial as spatial


#%% define the good channels and write it to a file

good_mics = [0,1,3,4,5,6,
             8,9,10,11,
             16,17,18,19,20,21,22]

with open('2018-08-19_goodmics.txt','w') as txtfile:
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
# These measurements are taken from the field notes.
def roundmean(x1x2):
    return np.round(np.mean(x1x2), 3)


channel_mic2mic = {}
channel_mic2mic['s9,1'] = roundmean([3.622, 3.607])
channel_mic2mic['s9,2'] = roundmean([3.549, 3.551])
channel_mic2mic['s9,4'] = 3.154
channel_mic2mic['s9,5'] = 3.119
channel_mic2mic['s9,6'] = roundmean([3.091, 3.088])
channel_mic2mic['s9,7'] = roundmean([3.358, 3.344])
channel_mic2mic['s9,2p'] = roundmean([3.303, 3.314])
channel_mic2mic['s9,3p'] = roundmean([3.270, 3.272])
channel_mic2mic['s9,4p'] = roundmean([3.168, 3.176])
channel_mic2mic['s9,5p'] = roundmean([3.067, 3.069])
channel_mic2mic['s9,6p'] = roundmean([2.851, 2.865])
channel_mic2mic['s9,7p'] = roundmean([2.294, 2.317])
channel_mic2mic['s9,8p'] = roundmean([1.956, 1.964])

# sanken 10 to smps

channel_mic2mic['s10,1'] = roundmean([4.008, 4.023])
channel_mic2mic['s10,2'] = roundmean([3.884, 3.895])
channel_mic2mic['s10,4'] = roundmean([3.899, 3.909])
channel_mic2mic['s10,5'] = roundmean([3.817, 3.822])
channel_mic2mic['s10,6'] = roundmean([3.684, 3.687])
channel_mic2mic['s10,7'] = roundmean([4.318, 4.325])
channel_mic2mic['s10,2p'] = roundmean([3.840, 3.837])
channel_mic2mic['s10,3p'] = roundmean([3.731, 3.735])
channel_mic2mic['s10,4p'] = roundmean([4.030, 4.019])
channel_mic2mic['s10,5p'] = roundmean([3.827, 3.824])
channel_mic2mic['s10,6p'] = roundmean([3.953, 3.959])
channel_mic2mic['s10,7p'] = roundmean([3.262, 3.267])

# sanken 11 to smps

channel_mic2mic['s11,1'] = 3.708
channel_mic2mic['s11,2'] = roundmean([3.593, 3.588])
channel_mic2mic['s11,4'] = roundmean([2.910, 2.894])
channel_mic2mic['s11,5'] = roundmean([2.818, 2.816])
channel_mic2mic['s11,6'] = roundmean([2.620])
channel_mic2mic['s11,7'] = roundmean([2.768, 2.760])
channel_mic2mic['s11,2p'] = roundmean([3.222, 3.224])
channel_mic2mic['s11,3p'] = roundmean([3.117, 3.111])
channel_mic2mic['s11,4p'] = roundmean([2.700, 2.699])
channel_mic2mic['s11,5p'] = roundmean([2.492, 2.492])
channel_mic2mic['s11,6p'] = roundmean([2.180, 2.120])
channel_mic2mic['s11,7p'] = roundmean([1.298, 1.278])
channel_mic2mic['s11,8p'] = roundmean([1.539, 1.545])


#%% Make the mapping between SMP mic number and the final channel number (0-indexed)
mic_to_channel = {'1':0,
                  '2':1,
                  '4':2,
                  '5':3,
                  '6':4,
                  '7':5,
                  's9':6,
                  's10':7,
                  's11':8,
                  's12':9,
                  '2p':10,
                  '3p':11,
                  '4p':12,
                  '5p':13,
                  '6p':14,
                  '7p':15,
                  '8p':16
                }

#%% Include the tristar distance matrix for the SANKEN9-12

mic2mic_distmat = np.empty((len(good_mics), len(good_mics)))
mic2mic_distmat[:] = np.nan
mic2mic_distmat[6:10,6:10] = tristar_distmat

for pair, distance in channel_mic2mic.items():
    smp_mica, smp_micb = pair.split(',')

    channel_mica = mic_to_channel[smp_mica]
    channel_micb = mic_to_channel[smp_micb]
    if not np.logical_or(channel_mica is None, channel_micb is None):
        mic2mic_distmat[channel_mica, channel_micb] = distance


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

# #%% 
distmat = pd.DataFrame(mic2mic_distmat)
distmat.columns = [ 'channel'+str(each+1) for each in range(len(good_mics))]
distmat.index = [ 'channel'+str(each+1) for each in range(len(good_mics))]

#%% Write the distance matrix:
distmat.to_csv('mic2mic_RAW_2018-08-19.csv',na_rep='NaN')
