# -*- coding: utf-8 -*-
"""
Visualing bat trajectories 2018-06-21 8000TMC - 310-335 frames
==============================================================

Created on Wed Apr 19 17:56:33 2023

@author: theja
"""
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

batxyz = pd.read_csv('22000/DLTdv7_data_2018-06-21_P00_22000_round1dltcoefs_xyzpts.csv')
numcols = batxyz.shape[1]
all_batxyz = []
for start in range(0,numcols,3):
    start_col, end_col = batxyz.columns[start], batxyz.columns[start+2]
    subdf = batxyz.loc[:,start_col:end_col].copy()
    subdf['batid'] = start_col.split('_')[0][2:]
    subdf['batid'] = start_col.split('_')[0][2:]
    subdf['frame'] = list(subdf.index)
    subdf.columns = ['x','y','z','batid','frame']
    all_batxyz.append(subdf)
all_batxyz = pd.concat(all_batxyz).reset_index(drop=True)

micxyz_df = pd.read_csv('DLTdv7_data_2018-06-21_mics_round1xyzpts.csv').dropna().reset_index(drop=True)
micxyz_df.columns = ['x', 'y', 'z']
current_order = ['SMP'+str(each) for each in range(1,9)] + ['SANKEN'+str(every) for every in [9,10,11,12]]
micxyz_df['micname'] = current_order
micxyz = micxyz_df.loc[:,'x':'z'].to_numpy()

#%%
plt.figure()
a0 = plt.subplot(111, projection='3d')
plt.plot(micxyz[:,0], micxyz[:,1], micxyz[:,2], '*')

for batid, subdf in all_batxyz.groupby('batid'):
    plt.plot(subdf.loc[:,'x'], subdf.loc[:,'y'], subdf.loc[:,'z'])

#%% 
# choose only those mics which are in the audio file: S9-12, SMP1-6
audio_order = current_order[-4:] + current_order[:6]
audio_order_inds = [np.where(micxyz_df['micname']==each)[0][0] for each in audio_order]
ordered_micxyz_df = micxyz_df.loc[audio_order_inds,:].reset_index(drop=True)
#%%
# Also try to find the rotation+translation matrix to go from camera coordinate system
# to TotalStation system
ordered_micxyz = ordered_micxyz_df.loc[:,'x':'z'].to_numpy()
totalstation_xyz_df = pd.read_csv('../../audio-video-matching/arraygeom_2018-06-21_1529543496.csv')
totalstation_xyz = totalstation_xyz_df.loc[:,'x':'z'].to_numpy()

totalstation_xyz_sanken9centred = totalstation_xyz - totalstation_xyz[0,:]
ordered_micxyz_sanken9centred = ordered_micxyz - ordered_micxyz[0,:]

from scipy.spatial.transform import Rotation
out, ssd = Rotation.align_vectors(totalstation_xyz_sanken9centred, ordered_micxyz_sanken9centred)
rotmat = out.as_matrix()
pd.DataFrame(data=rotmat).to_csv('Sanken9_centred-video-to-TotalStation_transform.csv')

ordered_micxyz_sanken9centred_df = pd.DataFrame(ordered_micxyz_sanken9centred)
ordered_micxyz_sanken9centred_df['micname'] = ordered_micxyz_df.loc[:,'micname']
ordered_micxyz_sanken9centred_df.to_csv('Sanken9_centred_mic_videoxyz.csv')

totalstation_xyz_sanken9centred_df = pd.DataFrame(totalstation_xyz_sanken9centred)
totalstation_xyz_sanken9centred_df['micname'] = ordered_micxyz_df.loc[:,'micname']
totalstation_xyz_sanken9centred_df.to_csv('Sanken9_centred_mic_totalstationxyz.csv')
#%% And now centre all the bat flight trajectories to SANKEN9 too. 
all_batxyz.loc[:,'x':'z'] -=  ordered_micxyz[0,:]
all_batxyz.to_csv('bat_trajs_round1_sanken9_centred.csv')

#%%
plt.figure()
a0 = plt.subplot(111, projection='3d')
plt.plot(ordered_micxyz_sanken9centred[:,0], ordered_micxyz_sanken9centred[:,1], ordered_micxyz_sanken9centred[:,2], '*')
plt.plot(ordered_micxyz_sanken9centred[0,0],
         ordered_micxyz_sanken9centred[0,1],
         ordered_micxyz_sanken9centred[0,2], 'k*')

for batid, subdf in all_batxyz.groupby('batid'):
    plt.plot(subdf.loc[:,'x'], subdf.loc[:,'y'], subdf.loc[:,'z'])
