# -*- coding: utf-8 -*-
"""
Creating mic2mic matrix for 2018-08-22
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


#%% define the good mics and write it to a file
good_mics = [0,1,2,3,4,5,
             9,10,11,12,13]

with open('2018-06-22_goodmics.txt','w') as txtfile:
    txtfile.write(str(good_mics)[1:-1])

#%% 
# Initiate the distance matrix with just channels 0-3 in it 
R = 1.2
theta = np.pi/3
mic00 = [0,0,0]
mic01 = [-R*np.sin(theta), 0, -R*np.cos(theta)]
mic02 = [R*np.sin(theta),  0, -R*np.cos(theta)]
mic03 = [0, 0, R]

tristar = np.row_stack((mic00, mic01, mic02, mic03))
tristar_distmat = spatial.distance_matrix(tristar, tristar)

#%% 

mic2mic_distmat = np.empty((len(good_mics), len(good_mics)))
mic2mic_distmat[:] = np.nan
mic2mic_distmat[:4,:4] = tristar_distmat

distmat = pd.DataFrame(mic2mic_distmat)
distmat.columns = [ 'channel'+str(each+1) for each in range(len(good_mics))]
distmat.index = [ 'channel'+str(each+1) for each in range(len(good_mics))]

#%% Write the distance matrix:
distmat.to_csv('mic2mic_RAW_2018-06-22.csv',na_rep='NaN')    
