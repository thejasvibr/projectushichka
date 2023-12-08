# -*- coding: utf-8 -*-
"""
Naive blob detection using connectivity only
--------------------------------------------
A simple blob detection method implemented only using connectivity
after thresholding. 

"""

import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import skimage 
from skimage import measure
import scipy.ndimage as ndi
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import disk
from skimage.filters import try_all_threshold, threshold_yen, threshold_triangle
from skimage.feature import blob_dog, blob_log, blob_doh
import trackpy as tp
import tqdm
dB = lambda X: 20*np.log10(np.abs(X))
#%%
folder = '2018-08-17/K3/P001/png/'
images = skimage.io.imread_collection(folder+'*.png')

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

#%%
substack = images[:]
cleaned_images = []
print('Cleaning fixed pattern noise ...\n')
for each in tqdm.tqdm(substack):
    cleaned_images.append(remove_vert_horiz_lines(each[:,:,2]))

#%%
# Doesn't work as well

invert = False
# allblobs = []
frame_nums = range(len(cleaned_images))
# for each in tqdm.tqdm(frame_nums):
#     tgt = cleaned_images[each]
#     if invert:
#         tgt = np.invert(tgt)
#     thresh = threshold_yen(tgt)
#     binzd = tgt >thresh
#     blobs, objects = ndi.label(binzd)
#     objects = ndi.find_objects(blobs)
#     allblobs.append(objects)

#%%

def process_object_data(detn):
    xl, yl = detn
    corner = (yl.start, xl.start)
    width = xl.stop-xl.start +1
    height = yl.stop-yl.start +1
    return corner, width, height
    
#%%
# visualise them as boxes

def place_bounding_box(image, rois):
    '''

    '''
    plt.figure()
    plt.imshow(image)    
    ax = plt.gca()
    for roi in rois:
        xy, width, height = process_object_data(roi)
        rectangle = plt.Rectangle(xy, width, height, fill=False, color='w')
        ax.add_artist(rectangle)

#%%
# There are many false positives being generated by small dots from the wings
# pass the original OR binarised image through a mean/median filter to 
# reduce the chance of such things happening?

print('Finding blobs...')
all_contours = []
all_regions = []
for each in tqdm.tqdm(frame_nums):
    tgt = cleaned_images[each]
    if invert:
        tgt = np.invert(tgt)
    thresh = threshold_yen(tgt, nbins=100)
    binzd = tgt >thresh
    regions = measure.label(np.uint8(binzd),)
    contours = measure.find_contours(np.float32(binzd), 0.9)
    all_contours.append(contours)
    all_regions.append(regions)

#%%
# Get region properties
msmts = []
for each in all_regions:
    msmts.append(measure.regionprops(each))

#%%

print('plotting detected blobs')

plt.ioff()
for idx,each in tqdm.tqdm(enumerate(substack)):
    
    plt.figure(figsize=(8,4))
    ax = plt.subplot(121)
    ax.imshow(each)
    
    for i,region in enumerate(msmts[idx]):
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red',
                                  linewidth=0.5)
        ax.add_patch(rect)
        plt.text(minc, minr,str(i),
                     fontsize=7)
    ax2 = plt.subplot(122)
    plt.imshow(each)
    plt.savefig(f'{idx}_skimage.png')
    plt.close()
 
plt.ion()