# -*- coding: utf-8 -*-
"""
Creating mic2mic matrix from TotalStation x,y,z measurements: 2018-08-21
========================================================================


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
good_mics = [0,1,2,3,
             4,5,9,10,11,12]
ids = ['S0','S1','S2','S3','M1','M2','M4','M5','M6','M7']

with open('2018-06-21_goodmics.txt','w') as txtfile:
    txtfile.write(str(good_mics)[1:-1])
#%% load Totalstation positions

df = pd.read_csv('Cave_w_channel_numbers.csv')
mic_pos = df[~pd.isna(df['channel_num'])]
#%%

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=16, azim=165)
ax.plot(mic_pos['X'], -mic_pos['Y'], mic_pos['Z'],'*')
ax.set_box_aspect(aspect = (1,1,1))

#%% 
subset_micpos = mic_pos[mic_pos['Object'].isin(ids)]
subset_micpos = subset_micpos.sort_values(by='channel_num')
xyz = subset_micpos.loc[:,['X','Y','Z']].to_numpy()
mic2mic_distmat = spatial.distance_matrix(xyz, xyz)
distmat = pd.DataFrame(mic2mic_distmat)
distmat.columns = [ 'channel'+str(each+1) for each in range(10)]
distmat.index = [ 'channel'+str(each+1) for each in range(10)]

#%% Write the distance matrix:
distmat.to_csv('mic2mic_RAW_2018-06-21.csv')    
