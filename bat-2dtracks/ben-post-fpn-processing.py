# -*- coding: utf-8 -*-
"""
Using Ben's code on the cleaned up images 
-----------------------------------------


"""
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import cv2
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from sklearn.linear_model import LogisticRegression
import skimage 
import natsort
from skimage import measure
from skimage.filters import threshold_yen
import tqdm
import trackpy as tp
import bat_functions as kbf
dB = lambda X: 20*np.log10(np.abs(X))
#%%
folder = '2018-08-17/K2/P001/png/'
image_paths = glob.glob(folder+'*7000*.png')[:200]
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
inverted_images = []
print('Inverting cleaned images')
for num, img in tqdm.tqdm(enumerate(cleaned_images)):
    threechannel = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(3):
        threechannel[:,:,i] = img
    inv_img = np.invert(np.uint8(threechannel))
    inverted_images.append(inv_img)
    skimage.io.imsave(f'cleaned_imgs/cleaned_{num}.png', inv_img)
inv_stack = np.array(inverted_images)
#%%
if not os.path.exists('ben_postfpn/'):
    os.mkdir('ben_postfpn/')
# make sure ben_postfpn is clean
for each in glob.glob('ben_postfpn/*'):
    os.remove(each)

camera_id = 'K3'
image_files = natsort.natsorted(glob.glob('cleaned_imgs/cleaned*.png'))
all_frame_data = []

bat_thresh = 0.1
bat_area = 1
print('Detecting bats .....')
for focal_frame_ind,_ in tqdm.tqdm(enumerate(image_files)):
    output = kbf.simple_process_frame(image_files, 
                                  bat_thresh, bat_area,
                                  focal_frame_ind)


    frame = cv2.imread(image_files[focal_frame_ind])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    circled_image = kbf.draw_circles_on_image(frame, 
                                          output['bat_centers'], 
                                          output['bat_sizes'], 
                                          rects=output['bat_rects'])

    # save color image with tracking info
    plt.imsave('ben_postfpn/'+f'{camera_id}_{focal_frame_ind}_tracked.png', circled_image)
    # save centers of object centres
    df = pd.DataFrame(output['bat_centers'],columns=['col_index','row_index'])
    df['camera_id'] = camera_id
    df['frame_number'] = focal_frame_ind+1
    df['file_name'] = os.path.split(image_files[focal_frame_ind])[-1]
    df['tracking_parameters'] = f'bat_thresh:{bat_thresh}&bat_area:{bat_area}'
    all_frame_data.append(df)

all_camera_data = pd.concat(all_frame_data).reset_index(drop=True)
all_camera_data = all_camera_data.rename(columns={'frame_number':'frame', 'col_index':'x','row_index':'y'})
#%%
# Now use trackpy to get the 

sr = 30; mem= 5;
tracks = tp.link(all_camera_data, search_range=sr,
                 memory=mem)

#%%
# No need to perform the nearest velocity predict - same/ext similar 
# type of tracking.


#%%
if not os.path.exists('tracks_overlaid/'):
    os.mkdir('tracks_overlaid')

def get_pastXframes(tracks, past_frames):
    ''' Generates moving window of a trajectory dataframe with 
    current frame till current-X frames
    '''
    past_10frames = num+1 -past_frames
    if past_10frames>0:
        frame_range = np.logical_and(tracks['frame'] <=num+1,
                                     tracks['frame']>past_10frames)      
    else:
        frame_range = tracks['frame'] <= num+1
    tracks_subset = tracks[frame_range]
    return tracks_subset

particles_byframe = all_camera_data.groupby(by='frame')

plt.figure()
plt.ioff()
for num, img in tqdm.tqdm(enumerate(cleaned_images)):
    thisframe = tracks[tracks['frame']==num+1]
    bats_thisframe = particles_byframe.get_group(num+1)
    tracks_subset = get_pastXframes(tracks, 5)
    a0 = plt.subplot(111);
    plt.title(f'normal frame:{num}')
    tp.plot_traj(tracks_subset, superimpose=np.invert(cleaned_images[num]), label=True)
    plt.plot(bats_thisframe['x'],  bats_thisframe['y'], 'r+', markersize=0.75)
    plt.savefig(f'tracks_overlaid/overlaid_{num}.png')
    a0.cla()
plt.close()

#%% The tracks are getting broken every now and then - why is this happening?
# Let's plot the detected bats over time. 



plt.figure()
plt.ioff()
for num, img in tqdm.tqdm(enumerate(cleaned_images)):
    thisframe = tracks[tracks['frame']==num+1]
    bats_thisframe = particles_byframe.get_group(num+1)
    tracks_subset = get_pastXframes(tracks, 15)
    rows = thisframe.shape[0]
    a0 = plt.subplot(121);
    plt.title(f'normal frame:{num}')
    tp.plot_traj(tracks_subset, superimpose=np.invert(cleaned_images[num]), label=True)
    a1= plt.subplot(122);
    plt.title('Only detected particles')
    plt.imshow(cleaned_images[num])
    plt.plot(bats_thisframe['x'],  bats_thisframe['y'], 'r+', markersize=0.75)
    plt.savefig(f'tracks_overlaid/tracks_particles_overlaid_{num}.png')
    a0.cla()
    a1.cla()
plt.close()




# #%% 

# if not os.path.exists('tracks_clean/'):
#     os.mkdir('tracks_clean')

# plt.ioff()
# for num, img in tqdm.tqdm(enumerate(cleaned_images)):
#     thisframe = tracks[tracks['frame']==num+1]
#     rows = thisframe.shape[0]
#     plt.figure()
#     plt.imshow(img)
#     ax = plt.gca()
#     for j, (x,y,_, frame, _, _, particle) in thisframe.iterrows():
#         plt.text(x,y,str(particle),fontsize=7)
#     plt.savefig(f'tracks_clean/trajnums_{num+1}.png')
# plt.ion()
        
        
    
    