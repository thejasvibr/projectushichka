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
import matplotlib.pyplot as plt
import numpy as np
import skimage 

from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import disk
from skimage.filters import try_all_threshold, threshold_yen, threshold_triangle
from skimage.feature import blob_dog, blob_log, blob_doh
import tqdm
dB = lambda X: 20*np.log10(np.abs(X))
#%%
images = skimage.io.imread_collection('2018-08-17/K1/P001/png/*inv*.png')

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
cleaned_images = []
for each in tqdm.tqdm(images):
    cleaned_images.append(remove_vert_horiz_lines(each[:,:,2]))
#%%
tgt = np.invert(cleaned_images[110])

fig, ax = try_all_threshold(tgt, figsize=(10, 8), verbose=False)
plt.show()
#%%
# 
invert = True
allblobs = []
frame_nums = range(len(cleaned_images))
for each in tqdm.tqdm(frame_nums):
    tgt = cleaned_images[each]
    if invert:
        tgt = np.invert(tgt)
    thresh = threshold_yen(tgt)
    binzd = tgt >thresh
    blobs = blob_doh(binzd, overlap=0.3)
    allblobs.append(blobs)
#%%
for i, blobs in enumerate(allblobs):
    plt.figure()
    plt.imshow(cleaned_images[i])
    ax = plt.gca()
    for each in blobs:
        c = plt.Circle((each[1], each[0]), 10, fill=False)
        ax.add_artist(c)
    plt.savefig(f'img_w_blobs_{i}.png')
    plt.close()