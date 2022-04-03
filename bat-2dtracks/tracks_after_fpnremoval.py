# -*- coding: utf-8 -*-
"""
Removing fixed pattern noise and performing 2D tracking 
-------------------------------------------------------
I've been having issues performing blob detection, perhaps because 
of the varying intensity of the fixed pattern noise appearing as
horizontal and vertical lines. 

The idea is to remove the horizontal and vertical lines by suppressing their
coefficients in a 2D FFT. This approach is based on Julian's MATLAB code
implemented in the removeFPNNoise function in 'main_01_format_raw.m'.

Overall Impression 
------------------
This method is working pretty damn well. Removing fixed pattern noise
and then performing blob detection helps a LOT!!!

References
----------
* Julian Jandeleit, Ushichka registration repo: https://gitlab.inf.uni-konstanz.de/julian.jandeleit/ushichka-registration
"""
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage 

from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import disk
from skimage.filters import try_all_threshold, threshold_yen, threshold_triangle
from skimage.feature import blob_dog, blob_log, blob_doh
import trackpy as tp
import tqdm
dB = lambda X: 20*np.log10(np.abs(X))
#%%
folder = '2018-08-17/K2/P001/png/'
images = skimage.io.imread_collection(folder+'*.png')[100:200]

#%%
img = images[13][:,:,2]
#% 

def remove_vert_horiz_lines(image):
    '''Specifically designed only for 640 x 512 images!!
    Thanks to roadrunner66 for the SO answer https://stackoverflow.com/a/37447304/4955732
    '''
    if image.shape != (512,640):
        raise ValueError(f'Only 512x640 images permitted. Current shape is {image.shape}')
    twod_fft = np.fft.fft2(image)
    shifted = np.fft.fftshift(twod_fft)
    
    clean_fft = shifted.copy()
    clean_fft[256,:] = 0
    clean_fft[:,320] = 0
    clean_img = np.abs(np.fft.ifft2(clean_fft))
    clean_img = np.array(clean_img, dtype=image.dtype)
    return clean_img

raw = img.copy()
cleaned = remove_vert_horiz_lines(raw)

plt.figure()
plt.imshow(cleaned)

#%% 
# Run cleaning on all images
substack = images[:]
cleaned_images = []
print('Cleaning fixed pattern noise ...\n')
for each in tqdm.tqdm(substack):
    cleaned_images.append(remove_vert_horiz_lines(each[:,:,2]))
# #%%
# tgt = np.invert(cleaned_images[10])

# fig, ax = try_all_threshold(tgt, figsize=(10, 8), verbose=False)
# plt.show()

#%%
# 
invert = False
allblobs = []
frame_nums = range(len(cleaned_images))
for each in tqdm.tqdm(frame_nums):
    tgt = cleaned_images[each]
    if invert:
        tgt = np.invert(tgt)
    thresh = threshold_yen(tgt)
    binzd = tgt >thresh
    blobs = blob_dog(binzd, overlap=0.2, 
                     min_sigma=0.25, max_sigma=20, threshold_rel=0.3)
    allblobs.append(blobs)

#%%
def plot_blobs(image, blobs):
    plt.figure()
    plt.imshow(image)
    ax = plt.gca()
    for each in blobs:
        c = plt.Circle((each[1], each[0]), each[2], fill=False)
        ax.add_artist(c)
print('Plotting blobs onto images...')
for i, blobs in tqdm.tqdm(enumerate(allblobs)):
    plt.ioff()
    plt.figure()
    plot_blobs(substack[i][:,:,2], blobs)
    plt.savefig(f'img_w_blobs_{i}.png')
    plt.close()
#%%
all_blob_data = []
for i,each in tqdm.tqdm(enumerate(allblobs)):
    blob_locs = pd.DataFrame(data={'x':[],'y':[],'frame':[]})
    blob_locs['x'] = each[:,1]
    blob_locs['y'] = each[:,0]
    blob_locs['frame'] = i
    all_blob_data.append(blob_locs)
allblob_data = pd.concat(all_blob_data).reset_index(drop=True)

#%% Link blob blocations into tracks
linked = tp.link(allblob_data, search_range=30, memory=5)
filt_linked = tp.filter_stubs(linked)
by_pid = filt_linked.groupby('particle')

#%%
# Remove all tracks which are suspiciously stationary
def startend_travelled(track_df):
    if track_df.shape[0] < 2:
        return 0
    
    start_pos = track_df.loc[np.min(track_df.index),['x','y']]
    end_pos = track_df.loc[np.max(track_df.index),['x','y']]
    dist = np.sqrt(np.sum((end_pos-start_pos)**2))
    return dist

pids = filt_linked['particle'].unique()

filt_linked['dist_travelled'] = np.nan
for each in pids:
    subdf_rows = filt_linked['particle']==each
    subdf = filt_linked[subdf_rows]
    distance = startend_travelled(subdf)
    filt_linked.loc[subdf_rows,'dist_travelled'] = distance

#%%
travel_threshold = 8
nonstat_tracks = filt_linked[filt_linked['dist_travelled']>travel_threshold].reset_index(drop=True)
pids =  nonstat_tracks['particle'].unique()

#%%
# plot trajectories for frames 1:10
max_frame_num = 99
plt.ion()
plt.figure()
plt.imshow(substack[max_frame_num][:,:,2])

frames_1to9 = nonstat_tracks[nonstat_tracks['frame']<=max_frame_num]
pids = frames_1to9['particle'].unique()
for each in pids:
    subdf = frames_1to9[frames_1to9['particle']==each]
    plt.plot(subdf['x'],subdf['y'],'.-')
    
#%%
cmap = plt.cm.get_cmap(plt.cm.rainbow_r, 50)
num_tracks = len(pids)
col_nums = np.linspace(0,50,num_tracks)
np.random.shuffle(col_nums)

for i, frame in tqdm.tqdm(enumerate(substack)):
    plt.ioff()
    plt.figure()
    ax = plt.gca()
    plt.imshow(frame[:,:,2])
    subdf = nonstat_tracks[nonstat_tracks['frame']==i]
    for row, (x,y,_,particle,_) in subdf.iterrows():
        ptcle_index = int(np.argwhere(pids==particle))
        #print(ptcle_index)
        ccl = plt.Circle((x,y),5,fill=False,
                         color=cmap(col_nums[ptcle_index]))
        plt.text(x+0.5, y+0.5, str(particle))
        ax.add_artist(ccl)
    plt.savefig(f'tracked_{i}_bats.png')
    plt.close()
    
    
