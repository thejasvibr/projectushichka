# -*- coding: utf-8 -*-
"""
Testing to see if the btrack package works well
-----------------------------------------------
Summary (2023-03-15): The `btrack` package works FANTASTICALLY - even with the package
default config parameters. I really couldn't be happier. Some experimenting
showed me that both 'VISUAL' and 'MOTION'  are required as input for the `tracking_updates`
parameter. Otherwise - I'm very impressed. Tracking is fairly flawless without much
problem. 


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
folder = '2018-08-17/K2/P001/png/'
image_paths = glob.glob(folder+'*7000*.png')[40:80]
images = [skimage.io.imread(each) for each in image_paths]
#images = skimage.io.imread_collection(folder+'*.png')

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
if not os.path.exists('cleaned_imgs/'):
    os.mkdir('cleaned_imgs')

# check that cleaned_imgs is empty
files = glob.glob('cleaned_imgs/*')
for f in files:
    os.remove(f)

footprint = disk(3)

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
    skimage.io.imsave(f'cleaned_imgs/cleaned_{num}.png', inv_img)
inv_stack = np.array(inverted_images)


#%%
import btrack
if not os.path.exists('btrack_segmentation/'):
    os.mkdir('btrack_segmentation/')
# make sure ben_postfpn is clean
for each in glob.glob('btrack_segmentation/*'):
    os.remove(each)

camera_id = folder.split('/')[1]
if len(camera_id)<1:
    raise ValueError('Invalid camera ID')
image_files = natsort.natsorted(glob.glob('cleaned_imgs/cleaned*.png'))
all_frame_data = []

#%%
if not os.path.exists('ben_postfpn/'):
    os.mkdir('ben_postfpn/')
# make sure ben_postfpn is clean
for each in glob.glob('ben_postfpn/*'):
    os.remove(each)

camera_id = folder.split('/')[1]
if len(camera_id)<1:
    raise ValueError('Invalid camera ID')
image_files = natsort.natsorted(glob.glob('cleaned_imgs/cleaned*.png'))
all_frame_data = []



bat_thresh = 0.05
bat_area = 0.075
print('Detecting bats .....')
all_masks = []
for focal_frame_ind,_ in tqdm.tqdm(enumerate(image_files)):
    output = kbf.simple_process_frame(image_files, 
                                  bat_thresh, bat_area,
                                  focal_frame_ind)
    mask = output['binary']
    all_masks.append(mask)

bat_detection_masks = np.array(all_masks)
#%%
FEATURES = ('area','solidity')
objects  = btrack.utils.segmentation_to_objects(bat_detection_masks,
                                                properties=FEATURES)    

CONFIG_FILE =  datasets.cell_config()

with btrack.BayesianTracker() as tracker:

   # configure the tracker using a config file
    tracker.configure(CONFIG_FILE)
    tracker.max_search_radius = 50
    tracker.tracking_updates = ["MOTION","VISUAL"]
    tracker.features = FEATURES

    # append the objects to be tracked
    tracker.append(objects)

    # set the tracking volume
    #tracker.volume=((0, 1600), (0, 1200))

    # track them (in interactive mode)
    tracker.track(step_size=25)

    # generate hypotheses and run the global optimizer
    tracker.optimize()

    # get the tracks in a format for napari visualization
    #data, properties, graph = tracker.to_napari()
    
    # store the tracks
    tracks = tracker.tracks
    
    # store the configuration
    cfg = tracker.configuration

#%%
from matplotlib.pyplot import cm
big_tracks = [each for each in tracks if len(each.t)>1]
color = cm.rainbow(np.linspace(0, 1, len(big_tracks)))


plt.figure()
plt.imshow(bat_detection_masks[0])
for i, bat_track in enumerate(big_tracks):
    plt.plot(bat_track.x, bat_track.y, c=color[i])
    plt.text(bat_track.x[0]-10, bat_track.y[0]+10, bat_track.ID, fontdict={'color':'white'})

#%%
# Save tracks and load onto pandas dataframe 
btrack.dataio.export_CSV('btrack_output.csv', tracks)
track_data = pd.read_csv('btrack_output.csv', delimiter=' ', index_col=False)

fig, a0 = plt.subplots()
by_frame = track_data.groupby('t')

for i, _ in enumerate(image_paths):
    plt.imshow(cleaned_images[i])
    frame_data = by_frame.get_group(i)
    if frame_data.shape[0]>0:
        plt.scatter(frame_data['x'], frame_data['y'], s=50,facecolors='none',
                    edgecolors='r', linewidths=0.2
                  )
    plt.savefig(f'particle_detections_frame_{i}.png')
    a0.cla()
plt.close()
#%%
# Save per trajectory


fig, a0 = plt.subplots()
by_frame = track_data.groupby('t')

col_by_traj = cm.rainbow(np.linspace(0, 1, len(track_data['ID'].unique())))
ids = track_data['ID'].unique()
by_traj = track_data.groupby('ID')

for i, _ in enumerate(image_paths):
    plt.imshow(cleaned_images[i])
    frame_data = by_frame.get_group(i)
    if frame_data.shape[0]>0:
        plt.scatter(frame_data['x'], frame_data['y'], s=50,facecolors='none',
                    edgecolors='r', linewidths=0.2
                  )
        for j, row in frame_data.iterrows():
            plt.text(row['x'], row['y'], str(int(row['ID'])), fontsize=10, color='w')
        
        for k, ID in enumerate(ids):
            this_particle = by_traj.get_group(ID)
            plt.plot(this_particle['x'], this_particle['y'], color=col_by_traj[k],
                     linewidth=0.5, linestyle='--')
            #plt.text(this_particle['x'])
    plt.title(f'frame {i}')
    plt.savefig(f'frame_{i}_detections.png')
    a0.cla()
plt.close()