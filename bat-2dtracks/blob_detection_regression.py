# -*- coding: utf-8 -*-
"""
Classifying blobs into bats or not
----------------------------------
Here I'll run a binary regression on each detected blob to 

@author: theja
"""
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import skimage 
from skimage import measure
from skimage.filters import threshold_yen
import tqdm
dB = lambda X: 20*np.log10(np.abs(X))
#%%
folder = 'blob_regression/images/'
images = skimage.io.imread_collection(folder+'*.png')

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
    
#%%

print('Finding blobs...')
all_contours = []
all_regions = []
for i, tgt in tqdm.tqdm(enumerate((cleaned_images))):
    thresh = threshold_yen(tgt, nbins=256)
    binzd = tgt >thresh*0.8
    regions = measure.label(np.uint8(binzd),)
    #contours = measure.find_contours(np.float32(binzd), 0.9)
    #all_contours.append(contours)
    all_regions.append(regions)

msmts = []
for each in all_regions:
    msmts.append(measure.regionprops(each))

#%%
def plot_raw_blobs(image, blob_msmts, ioff=False):
    plt.figure(figsize=(8,4))
    ax = plt.subplot(121)
    ax.imshow(image)    
    for i,region in enumerate(blob_msmts):
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red',
                                  linewidth=0.5)
        ax.add_patch(rect)
        plt.text(minc, minr,str(i),
                     fontsize=10, color='w')
    ax2 = plt.subplot(122, sharex=ax, sharey=ax)
    plt.imshow(image)




#%%
# Convert all the measurements into a dataframe

#%%
# Run a logistic regression OR some other kind of classifier
# Circularity, eccentricity and area are likely to play a big role - and be
# independent of how the image is exported. 
# Also mean region pixel value (sum pixels/num pixels) - rel the mean of the whole
# image.(use msmt.coords to get all region pixels)


input_data = pd.DataFrame(data={'file_name':[],
                                'blob_label':[],
                                'unique_label':[],
                                'area':[],
                                'mean_pixel':[],
                                'mean_re_image':[],
                                'perimeter':[],
                                'ecc':[]})

for input_image