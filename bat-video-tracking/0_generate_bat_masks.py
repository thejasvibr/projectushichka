# -*- coding: utf-8 -*-
"""
Creating bat detection masks for 2D tracking and linking
========================================================
Here we generate binary masks where 1's indicate bats in the frame.

Called by
~~~~~~~~~
* 20_overall_workflow.py

Modules to run after this
~~~~~~~~~~~~~~~~~~~~~~~~~
* btrack_checkingout.py
"""
import argparse
from difflib import SequenceMatcher
import glob
import numpy as np
import os
import pathlib
import skimage 
import natsort
import pandas as pd
from scipy import ndimage 
from skimage.filters import rank, threshold_yen
from skimage.morphology import disk
import tqdm

detailed_description = """Run bat detection by cleaning up thermal images, background subtraction, and thresholding followed\
                 by binarisation. Outputs are 1) cleaned and post-threshold masks\
    2) "*-2D-detections" csv file.Both images and the csv file have the common portion of the raw image file names\
    in them, followed by a serial number that always starts with 0.
    \
    \
    For example ```python 0_generate_bat_masks.py -source raw_images\\2018-08-18\\K1\\ -dest processed_images \
        -start 0 -end 24 -cam_id K1```    """

parser = argparse.ArgumentParser(description = detailed_description)
parser.add_argument('-source', type=str, help='Source folder with file pattern for all images. e.g.\
                     /images/K1/K1*.png . Input needs to be glob-able.')
parser.add_argument('-dest', type=str, default='./', help='Destination for all the cleaned camera images. \
                    Defaults to current folder. Folder will be made if it does not exist')
parser.add_argument('-start', type=int, default=0, help='Image number to start from. Defaults to 0')
parser.add_argument('-end', type=int, default=-1, help='Image number to end on. This number includes the frame\
                     Defaults to -1 (last one)')
parser.add_argument('-cam_id', type=str, help='Camera ID')
args = parser.parse_args()


#%%
cam_id = args.cam_id
image_paths = natsort.natsorted(glob.glob(args.source)[args.start:args.end+1])
images = skimage.io.imread_collection(image_paths)
#%%

a, b = [os.path.split(each)[-1] for each in [image_paths[0], image_paths[-1]]]
match = SequenceMatcher(None, a, b, autojunk=False)
common = match.find_longest_match(0,len(a),0,len(b))
common_str = a[:common.size]
print(f'Starting image cleaning and mask creation for {args.cam_id}: {common_str} image series')

def make_yenthreshold_binary_mask(X):
    ''' Makes a binary mask using the Yen thresholding
    Takes in a 2D np.array and outputs a boolean 2D np.array
    of the same size/shape.
    '''
    threshold = threshold_yen(X)
    return X>=threshold

def remove_vert_horiz_lines(image):
    '''Specifically designed only for 640 x 512 images!!
    Thanks to roadrunner66 for the SO answer https://stackoverflow.com/a/37447304/4955732
    Removes the fixed pattern noise (vertical and horizontal lines)
    in the thermal camera images.
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
dest_folder = pathlib.Path(f'{args.dest}/cleaned/{args.cam_id}/')
if not os.path.exists(dest_folder):
    dest_folder.mkdir(parents=True, exist_ok=True)

footprint = disk(2)

smoothed_stack = []
print('Smoothing cleaned images')
for num, img in tqdm.tqdm(enumerate(cleaned_images)):
    filtered_img = rank.mean(img, footprint=footprint)
    smoothed_stack.append(filtered_img)
    skimage.io.imsave(os.path.join(dest_folder, f'cleaned_{common_str}_{num}.png'), filtered_img)
smoothed_stack = np.array(smoothed_stack)

#%% 
# Generate masks by background subtraction. 
background_image = np.apply_along_axis(np.median, 0, smoothed_stack)
bground_subtr = smoothed_stack - background_image

#%%
masks_folder = pathlib.Path(os.path.join(dest_folder.parent.parent, fr'masks\{cam_id}/'))
if not os.path.exists(masks_folder):
    masks_folder.mkdir( parents=True, exist_ok=True)

bat_detection_masks = np.zeros(bground_subtr.shape, dtype=np.uint8)
bat_centers_byframe = []
for img in range(bground_subtr.shape[0]):
    bat_detection_masks[img,:,:] = make_yenthreshold_binary_mask(bground_subtr[img,:,:])
    mask_img_names = os.path.join(masks_folder, f'mask_{common_str}_{str(img)}.png')
    skimage.io.imsave(mask_img_names, bat_detection_masks[img,:,:]*255)
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




