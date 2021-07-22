# -*- coding: utf-8 -*-
"""
Creating mic2mic matrix for 2018-08-17
======================================
Session with 28 channel files

The order of microphone ids :
    * SMP1,2,3,4,5,6,7, SANKEN9,10,11,12, SMP8, SMP1',2',3',4',5',6',7',8'

Microphones SMP8 and SMP1' aren't included because the audio is digitised
thorugh the Focusrite OctoPre Scarlett, and the ADC delay is now known. 

ADAT channels not working
-------------------------
Field notes and the raw recordings indicate none of the ADAT channels 
that were working on 2018-08-14 were recording any data for some unknown reason -
perhaps due to low voltage of the batteries. 

Either way the ADAT channels need to be removed like on 2018-08-14, and 
so it doesn't make much of a difference

Like on 2018-08-14, it means SMP8 and SMP1' will be out of the final audio file. 

@author: tbeleyur
"""


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import scipy.spatial as spatial


#%% define the good channels and write it to a file
good_mics = [0,1,2,3,4,5,6,
             8,9,10,11,
             16,17,18,19,20,21,22]

with open('2018-08-17_goodmics.txt','w') as txtfile:
    txtfile.write(str(good_mics)[1:-1])

#%% 
# Initiate the distance matrix with just channels 0-3 in it 
#  The notes say that the order of Sankens9,10 were switched from the 
# standard - but I can't verify it on the photos - and so I'd rather leave 
# this info uncertain. 

R = 1.2
theta = np.pi/3
mic00 = [0,0,0]
mic01 = [999,999,999]# make it large to then replace with np.nan
mic02 = [999,999,999] # make it large to then replace with np.nan
mic03 = [0, 0, R]

tristar = np.row_stack((mic00, mic01, mic02, mic03))
tristar_distmat = spatial.distance_matrix(tristar, tristar)
tristar_distmat[tristar_distmat>3] = np.nan


#%% This refers to the MICROPHONE identitites, and not the channel IDs
# These measurements are taken from the field notes.
def roundmean(x1x2):
    return np.round(np.mean(x1x2), 3)

smp_mic2mic = {}
smp_mic2mic['8p,2'] = roundmean([4.248, 4.253])
smp_mic2mic['8p,3'] = roundmean([4.375, 4.365])
smp_mic2mic['8p,4'] = roundmean([3.174, 3.183])
smp_mic2mic['8p,5'] = roundmean([3.260, 3.241])
smp_mic2mic['8p,6'] = roundmean([3.334, 3.345])
smp_mic2mic['8p,7'] = roundmean([2.680, 2.681])
smp_mic2mic['8p,2p'] = roundmean([3.749, 3.758])
smp_mic2mic['8p,5p'] = roundmean([2.972, 2.975])
smp_mic2mic['8p,6p'] = roundmean([1.760, 1.775])
smp_mic2mic['7p,1'] = roundmean([3.444, 3.476])
smp_mic2mic['7p,2'] = roundmean([3.374, 3.397])
smp_mic2mic['7p,3'] = roundmean([3.306, 3.309])
smp_mic2mic['7p,4'] = roundmean([2.327, 2.336])
smp_mic2mic['7p,5'] = roundmean([2.232, 2.234])
smp_mic2mic['7p,6'] = roundmean([2.043, 2.048])
smp_mic2mic['7p,7'] = roundmean([1.736, 1.748])
smp_mic2mic['7p,2p'] = roundmean([2.886, 2.866])
smp_mic2mic['7p,3p'] = roundmean([2.774, 2.771])
smp_mic2mic['7p,4p'] = roundmean([1.876, 1.895])
smp_mic2mic['7p,5p'] = roundmean([1.683, 1.684])
smp_mic2mic['7p,s10'] = roundmean([1.271, 1.256])
smp_mic2mic['7p,s12'] = roundmean([2.832, 2.836])
smp_mic2mic['6p,1'] = roundmean([3.195, 3.263])
smp_mic2mic['6p,2'] = roundmean([3.314, 3.30])
smp_mic2mic['6p,3'] = roundmean([3.409, 3.434])
smp_mic2mic['6p,4'] = roundmean([1.897, 1.896])
smp_mic2mic['6p,6'] = roundmean([2.210, 2.149])
smp_mic2mic['6p,7'] = roundmean([0.978, 0.973])
smp_mic2mic['6p,2p'] = roundmean([2.661, 2.675])
smp_mic2mic['6p,3p'] = roundmean([2.722, 2.760])
smp_mic2mic['6p,4p'] = roundmean([1.423, 1.436])
smp_mic2mic['6p,5p'] = roundmean([1.616, 1.606])
smp_mic2mic['6p,s12'] = roundmean([2.904, 2.934])
smp_mic2mic['6p,s11'] = roundmean([2.159, 2.164])


#%% Some more SMP-SANKEN 9 distances (SANKEN9 is the 7th index channel (0index)):
smp_mic2mic['1,s9'] = np.round(np.mean([3.638,3.632]), 3)
smp_mic2mic['2,s9'] = np.round(np.mean([3.556, 3.549]), 3)
smp_mic2mic['3,s9'] = np.round(np.mean([3.556, 3.552]), 3)
smp_mic2mic['4,s9'] = np.round(np.mean([3.180, 3.174]), 3)
smp_mic2mic['5,s9'] = np.round(np.mean([3.134, 3.137]), 3)
smp_mic2mic['6,s9'] = np.round(np.mean([3.085, 3.096]), 3)
smp_mic2mic['7,s9'] = np.round(np.mean([3.370, 3.376]), 3)
smp_mic2mic['2p,s9'] = np.round(np.mean([3.333, 3.341]), 3)
smp_mic2mic['3p,s9'] = np.round(np.mean([3.288, 3.281]), 3)
smp_mic2mic['4p,s9'] = np.round(np.mean([3.196, 3.186]), 3)
smp_mic2mic['5p,s9'] = np.round(np.mean([3.076, 3.082]), 3)
smp_mic2mic['6p,s9'] = np.round(np.mean([2.864, 2.884]), 3)
smp_mic2mic['7p,s9'] = np.round(np.mean([2.350, 2.314]), 3)
smp_mic2mic['8p,s9'] = np.round(np.mean([1.976, 1.986]), 3)

#%% Make the mapping between SMP mic number and the final channel number (0-indexed)
micid_to_channel = {'1':0,
                  '2':1,
                  '3':2,
                  '4':3,
                  '5':4,
                  '6':5,
                  '7':6,
                  's9':7,
                  's10':8,
                  's11':9,
                  's12':10,
                  '2p':11,
                  '3p':12,
                  '4p':13,
                  '5p':14,
                  '6p':15,
                  '7p':16,
                  '8p':17,
                }

#%% Include the tristar distance matrix for the SANKEN9-12

mic2mic_distmat = np.empty((len(good_mics), len(good_mics)))
mic2mic_distmat[:] = np.nan
mic2mic_distmat[7:11,7:11] = tristar_distmat

for pair, distance in smp_mic2mic.items():
    smp_mica, smp_micb = pair.split(',')

    channel_mica = micid_to_channel[smp_mica]
    channel_micb = micid_to_channel[smp_micb]
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

#%% 
distmat = pd.DataFrame(mic2mic_distmat)
distmat.columns = [ 'channel'+str(each+1) for each in range(len(good_mics))]
distmat.index = [ 'channel'+str(each+1) for each in range(len(good_mics))]

#%% Write the distance matrix -- commented out for consistency
# distmat.to_csv('mic2mic_RAW_2018-08-17.csv',na_rep='NaN')

#%% Also write a smaller distance matrix - with only the fireface802 microphones 
ff802_mic2mic = distmat.loc[:'channel11',:'channel11']
ff802_mic2mic.to_csv('mic2mic_RAW_fireface802_2018-08-17.csv')



