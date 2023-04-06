# -*- coding: utf-8 -*-
"""
Creating bat detection masks for 2D tracking and linking
========================================================
Here we generate binary masks where 1's indicate bats in the frame.

Modules to run after this
~~~~~~~~~~~~~~~~~~~~~~~~~
btrack_checkingout.py
"""
import glob
import numpy as np
import os
import skimage 
import natsort
import pandas as pd
from scipy import ndimage 
from skimage.filters import rank, threshold_yen
from skimage.morphology import disk
import tqdm
dB = lambda X: 20*np.log10(np.abs(X))

#%%
cam_id = 'K1'
folder = f'2018-08-17/{cam_id}/P001/png/'
image_paths = natsort.natsorted(glob.glob(folder+'*7000*.png')[:25])
images = skimage.io.imread_collection(image_paths)
camera_id = folder.split('/')[1]
#%%

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
cleaned_images = []
print('Cleaning fixed pattern noise ...\n')
for each in tqdm.tqdm(images):
    cleaned_images.append(remove_vert_horiz_lines(each[:,:,2]))

minmax = [(np.min(X), np.max(X)) for X in cleaned_images]
#%%
if not os.path.exists(f'cleaned_imgs/{camera_id}/'):
    os.mkdir(f'cleaned_imgs/{camera_id}/')

# check that cleaned_imgs is empty
files = glob.glob(f'cleaned_imgs/{camera_id}/*')
for f in files:
    os.remove(f)

footprint = disk(2)

smoothed_stack = []
print('Smoothing cleaned images')
for num, img in tqdm.tqdm(enumerate(cleaned_images)):
    filtered_img = rank.mean(img, footprint=footprint)
    smoothed_stack.append(filtered_img)
    skimage.io.imsave(f'cleaned_imgs/{camera_id}/cleaned_{num}.png', filtered_img)
smoothed_stack = np.array(smoothed_stack)

#%%
def make_yenthreshold_binary_mask(X):
    ''' Makes a binary mask using the Yen thresholding'''
    threshold = threshold_yen(X)
    return X>=threshold
#%% 
# Generate masks by background subtraction. 
background_image = np.apply_along_axis(np.median, 0, smoothed_stack)
bground_subtr = smoothed_stack - background_image

#%%
from difflib import SequenceMatcher
a, b = [os.path.split(each)[-1] for each in [image_paths[0], image_paths[-1]]]
match = SequenceMatcher(None, a, b, autojunk=False)
common = match.find_longest_match(0,len(a),0,len(b))
common_str = a[:common.size]
if not os.path.exists('detection_masks'):
    os.mkdir('detection_masks')

bat_detection_masks = np.zeros(bground_subtr.shape, dtype=np.uint8)
bat_centers_byframe = []
for img in range(bground_subtr.shape[0]):
    bat_detection_masks[img,:,:] = make_yenthreshold_binary_mask(bground_subtr[img,:,:])
    skimage.io.imsave('detection_masks/'+common_str+str(img)+'_mask.png', bat_detection_masks[img,:,:]*255)
    labelled, num_bats =  ndimage.label(bat_detection_masks[img,:,:])
    detections = ndimage.find_objects(labelled)
    bat_centers = []
    for bat_detn in detections:
        bat_segment = bground_subtr[img,:,:][bat_detn]
        center_mass = ndimage.center_of_mass(bat_segment) # the row, column format output
        center_mass_global = (center_mass[0]+bat_detn[0].start, center_mass[1]+bat_detn[1].start)
        bat_centers.append(center_mass_global)
    bat_centers_df = pd.DataFrame(bat_centers, columns=['row','col'])
    bat_centers_df['frame'] = img
    bat_centers_byframe.append(bat_centers_df)

# Also save the 2D centres of all detected patches
bat_2D_positions = pd.concat(bat_centers_byframe).reset_index(drop=True)
# Save the positions with the common image file name + 'bat-2D-detections'
bat_detn_positions_filename = a[:common.size]+'-2D-detections.csv'
bat_2D_positions.to_csv(bat_detn_positions_filename)




