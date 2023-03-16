# -*- coding: utf-8 -*-
"""
Testing to see if the btrack package works well
-----------------------------------------------
Summary (2023-03-15): The `btrack` package works FANTASTICALLY - even with the package
default config parameters. I really couldn't be happier. Some experimenting
showed me that both 'VISUAL' and 'MOTION'  are required as input for the `tracking_updates`
parameter. Otherwise - I'm very impressed. Tracking is fairly flawless without much
problem. 


Things to check while diagnosing a track (the `napari` package already does this - whew!!)
> Plot track of interest in a diff color as the other detected tracks across frames
> Check to see if there are any sudden jumps. 
> Plot the tracks 

"""
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import cv2
import pandas as pd
import skimage 
import natsort
from skimage import measure
from skimage.filters import threshold_yen
import tqdm
dB = lambda X: 20*np.log10(np.abs(X))
import bat_functions as kbf
import btrack
from btrack import datasets
from skimage.filters import rank
from skimage.morphology import disk 

#%%
folder = '2018-08-17/K3/P001/png/'
image_paths = glob.glob(folder+'*7000*.png')[:25]
images = [skimage.io.imread(each) for each in image_paths]
#images = skimage.io.imread_collection(folder+'*.png')
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

inverted_images = []
print('Inverting cleaned images')
for num, img in tqdm.tqdm(enumerate(cleaned_images)):
    threechannel = np.zeros((img.shape[0], img.shape[1], 3))

    filtered_img = rank.mean(img, footprint=footprint)
    for i in range(3):
        threechannel[:,:,i] = filtered_img
    inv_img = np.invert(np.uint8(threechannel))
    # also smooth the images a bit - and check to see if it makes a difference
    
    inverted_images.append(inv_img)
    skimage.io.imsave(f'cleaned_imgs/{camera_id}/cleaned_{num}.png', inv_img)
inv_stack = np.array(inverted_images)


#%%
import btrack

if len(camera_id)<1:
    raise ValueError('Invalid camera ID')
image_files = natsort.natsorted(glob.glob(f'cleaned_imgs/{camera_id}/cleaned*.png'))
all_frame_data = []

#%%
if not os.path.exists(f'ben_postfpn/{camera_id}/'):
    os.mkdir(f'ben_postfpn/{camera_id}/')
# make sure ben_postfpn is clean
for each in glob.glob(f'ben_postfpn/{camera_id}/*'):
    os.remove(each)

camera_id = folder.split('/')[1]
if len(camera_id)<1:
    raise ValueError('Invalid camera ID')
image_files = natsort.natsorted(glob.glob(f'cleaned_imgs/{camera_id}/cleaned*.png'))
all_frame_data = []



bat_thresh = 0.05
bat_area = 0.08
print('Detecting bats .....')
all_masks = []
for focal_frame_ind,_ in tqdm.tqdm(enumerate(image_files)):
    output = kbf.simple_process_frame(image_files, 
                                  bat_thresh, bat_area,
                                  focal_frame_ind)
    mask = output['binary']
    all_masks.append(mask)
    plt.imsave(f'ben_postfpn/{camera_id}/{camera_id}_{focal_frame_ind}_mask.png', mask)

bat_detection_masks = np.array(all_masks)
#%%
FEATURES = ('area','axis_major_length','axis_minor_length','orientation',
            'solidity', 'eccentricity')

objects  = btrack.utils.segmentation_to_objects(bat_detection_masks,
                                                properties=FEATURES)    

CONFIG_FILE =  datasets.cell_config()
import napari

with btrack.BayesianTracker() as tracker:

   # configure the tracker using a config file
    tracker.configure(CONFIG_FILE)
    tracker.max_search_radius = 125
    tracker.tracking_updates = ["MOTION","VISUAL"]
    tracker.features = FEATURES

    # append the objects to be tracked
    tracker.append(objects)

    # set the tracking volume
    tracker.volume=((0, 1600), (0, 1200))

    # track them (in interactive mode)
    tracker.track(step_size=25)

    # generate hypotheses and run the global optimizer
    tracker.optimize()

    # get the tracks in a format for napari visualization
    data, properties, graph = tracker.to_napari()
    
    # store the tracks
    tracks = tracker.tracks
    
    # store the configuration
    cfg = tracker.configuration
#%%


viewer = napari.Viewer()

viewer.add_image(
    np.array(cleaned_images), 
    name="cleaned",
    opacity=0.9,
)

viewer.add_image(np.array(all_masks), 
                 name='mask',
                 opacity=0.2)
# the track data from the tracker
viewer.add_tracks(
    data, 
    properties=properties, 
    name="Tracks", 
    blending="translucent",
)
tracking_data = pd.DataFrame(data, columns=['id','frame','x','y'])
tracking_data.to_csv(f'{camera_id}_first25_tracks.csv')