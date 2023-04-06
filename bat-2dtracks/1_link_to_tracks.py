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

Modules to run before this
~~~~~~~~~~~~~~~~~~~~~~~~~~
* 0_generate_bat_masks.py

Modules to run after this
~~~~~~~~~~~~~~~~~~~~~~~~~
* traj_correction.py

"""
import argparse
import btrack
import glob
import napari
import numpy as np
import os 
import pandas as pd
import skimage 
import natsort
from difflib import SequenceMatcher as SeqMatch



#%%
cam_id = 'K3'
relevant_images = natsort.natsorted(glob.glob(f'detection_masks/{cam_id}*.png'))
bat_detection_masks = np.array([skimage.io.imread(each) for each in relevant_images])

# get common filename patterns
a,b = [os.path.split(each)[-1] for each in [relevant_images[0], relevant_images[-1]]]
common_range = SeqMatch(None, a, b, autojunk=False).find_longest_match(0,len(a),0,len(b))
common_str = a[:common_range.size]

#%%
FEATURES = ('area','axis_major_length','axis_minor_length','orientation',
             'solidity', 'eccentricity')

#FEATURES = ('area','orientation')

objects  = btrack.utils.segmentation_to_objects(bat_detection_masks,
                                                properties=FEATURES)    

CONFIG_FILE = 'bat_config.json'


with btrack.BayesianTracker() as tracker:

   # configure the tracker using a config file
    tracker.configure(CONFIG_FILE)
    tracker.max_search_radius = 100
    tracker.tracking_updates = ["MOTION","VISUAL"]
    tracker.features = FEATURES

    # append the objects to be tracked
    tracker.append(objects)

    # set the tracking volume
    tracker.volume=((0, 640), (0, 512)) # the X and Y volume to track!!

    # track them (in interactive mode)
    tracker.track(step_size=12)

    # generate hypotheses and run the global optimizer
    tracker.optimize()

    # get the tracks in a format for napari visualization
    data, properties, graph = tracker.to_napari()
    
    # store the tracks
    tracks = tracker.tracks
    
    # store the configuration
    cfg = tracker.configuration
#%%
cleaned_images = np.array(skimage.io.imread_collection(f'cleaned_imgs/{cam_id}/*.png'))

viewer = napari.Viewer()

viewer.add_image(
    np.array(cleaned_images), 
    name="cleaned",
    opacity=0.9,
)

viewer.add_image(bat_detection_masks, 
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
tracking_data.to_csv(f'{common_str}_tracks_first25_frames.csv')