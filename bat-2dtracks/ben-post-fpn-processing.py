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

from skimage.filters import rank
from skimage.morphology import disk 

#%%
folder = '2018-08-17/K3/P001/png/'
image_paths = glob.glob(folder+'*7000*.png')[:100]
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

bat_thresh = 0.05
bat_area = 0.5
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

sr = 60; mem= 5;
# tracks = tp.link(all_camera_data, search_range=sr,
#                  memory=mem)

pred = tp.predict.NearestVelocityPredict()
tracks = pred.link_df(all_camera_data, sr)

#%%
# No need to perform the nearest velocity predict - same/ext similar 
# type of tracking.


#%%
if not os.path.exists('tracks_overlaid/'):
    os.mkdir('tracks_overlaid')

def get_pastXframes(tracks, framenum, past_frames):
    ''' Generates moving window of a trajectory dataframe with 
    current frame till current-X frames
    '''
    past_10frames = framenum+1 -past_frames
    if past_10frames>0:
        frame_range = np.logical_and(tracks['frame'] <=framenum+1,
                                     tracks['frame']>past_10frames)      
    else:
        frame_range = tracks['frame'] <= framenum+1
    tracks_subset = tracks[frame_range]
    return tracks_subset

particles_byframe = all_camera_data.groupby(by='frame')

print('plotting trailing trajectories...')
max_pastframes = 5
plt.figure()
plt.ioff()
for num, img in tqdm.tqdm(enumerate(cleaned_images)):
    thisframe = tracks[tracks['frame']==num+1]
    bats_thisframe = particles_byframe.get_group(num+1)
    tracks_subset = get_pastXframes(tracks, num+1,max_pastframes)
    a0 = plt.subplot(111);
    plt.title(f'normal frame:{num}')
    tp.plot_traj(tracks_subset, superimpose=np.invert(cleaned_images[num]),
                 label=True, fontsize=3)
    plt.plot(bats_thisframe['x'],  bats_thisframe['y'], 'r+', markersize=0.75)
    plt.savefig(f'tracks_overlaid/overlaid_{num}.png')
    a0.cla()
plt.close()

#%% The tracks are getting broken every now and then - why is this happening?
# Let's plot the detected bats over time. 



plt.figure()
print('plotting single trajectories')
plt.ioff()
for num, img in tqdm.tqdm(enumerate(cleaned_images)):
    thisframe = tracks[tracks['frame']==num+1]
    bats_thisframe = particles_byframe.get_group(num+1)
    tracks_subset = get_pastXframes(tracks, num, max_pastframes)
    rows = thisframe.shape[0]
    a0 = plt.subplot(121);
    plt.title(f'normal frame:{num}')
    tp.plot_traj(tracks_subset, superimpose=np.invert(cleaned_images[num]),
                 label=True)
    a1= plt.subplot(122);
    plt.title('Only detected particles')
    plt.imshow(cleaned_images[num])
    a1.scatter(bats_thisframe['x'],  bats_thisframe['y'], s=50,facecolors='none',
               edgecolors='r', linewidths=0.2
             )

    for i, row in thisframe.iterrows():
        plt.text(row['x'], row['y'], row['particle'], fontsize=8)
    
    plt.savefig(f'tracks_overlaid/tracks_particles_overlaid_{num}.png')
    a0.cla()
    a1.cla()
plt.close()

#%%
        
plt.figure()
a0 = plt.gca()
tp.plot_traj(tracks, superimpose=inv_stack[-1,:,:,0], ax=a0,
             label=True,  colorby='particle', fontsize=0.4, 
             plot_style={'linewidth':0.5})
plt.show()
#%%
all_particle_ids = tracks.groupby('particle').groups.keys()
tracks_grouped = tracks.groupby('particle')


plt.figure()
a0 = plt.subplot(111)
for particle in tqdm.tqdm(all_particle_ids):
    this_traj = tracks_grouped.get_group(particle)
    if not os.path.exists(f'particle_{particle}_track/'):
        os.mkdir(f'particle_{particle}_track/')
    for i, row in this_traj.iterrows():
        frame = row['frame']
        x,y = row['x'], row['y']
        traj_trail = get_pastXframes(this_traj, frame, 5)
        plt.title(f'particle ID : {particle}, frame: {frame}')
        a0.scatter(x,y, s=50, facecolors='none', edgecolors='r', linewidths=0.1)
        plt.plot(traj_trail['x'], traj_trail['y'], 'w', linewidth=0.5)
        plt.imshow(cleaned_images[frame-1])
        plt.savefig(f'particle_{particle}_track/particle_{particle}_frame-{frame}.png')
        a0.cla()
plt.close()

#%%
from matplotlib.widgets import RangeSlider, CheckButtons

frames = range(len(cleaned_images))
fig, a0 = plt.subplots()

s1_ax = fig.add_axes([0.2,0.05,0.65,0.03])
minmax_frame = RangeSlider(ax=s1_ax, valmin=min(frames), valmax=max(frames), 
                           label='start frame', valinit=(min(frames), max(frames)),
                           valstep=np.arange(min(frames),max(frames)+1))

trajs_ax = fig.add_axes([0.8, 0.1, 0.2, 0.85])
all_particles = tracks['particle'].unique().tolist()
traj_buttons = CheckButtons(ax= trajs_ax, labels=all_particles)

for r in traj_buttons.rectangles:
    r.set_width(r.get_height()+0.01)
    r.set_height(r.get_height()+0.01)


for i,_ in enumerate(all_particles):
    traj_buttons.set_active(i)

plt.sca(a0)
plt.imshow(cleaned_images[-1])
tp.plot_
def update(val):
    plt.sca(a0)
    
    print('hi')
    checked_particles = traj_buttons.get_status()
    displayed_particles = []
    for active, particle in zip(checked_particles, all_particles):
        if active:
            displayed_particles.append(particle)

    dataset = tracks[np.logical_and(tracks['frame']>=minmax_frame.val[0],
                                    tracks['frame']<=minmax_frame.val[1])]
    filt_dataset = dataset.loc[dataset['particle'].isin(displayed_particles)]
    a0.cla()
    for i, checked in enumerate(checked_particles):
        if not checked:
            print('NOT', all_particles[i])

    tp.plot_traj(filt_dataset, ax=a0, superimpose=inv_stack[int(minmax_frame.val[1])-1,:,:,2], 
                 label=True)

minmax_frame.on_changed(update)
traj_buttons.on_clicked(update)


plt.show()