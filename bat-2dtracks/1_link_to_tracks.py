# -*- coding: utf-8 -*-
"""
Generate 2D tracks by linking object detections across binary masks
-------------------------------------------------------------------
Summary (2023-03-15): The `btrack` package works FANTASTICALLY - even with the package
default config parameters. I really couldn't be happier. Some experimenting
showed me that both 'VISUAL' and 'MOTION'  are required as input for the `tracking_updates`
parameter. Otherwise - I'm very impressed. Tracking is fairly flawless without much
problem. 


Things to check while diagnosing a track (the `napari` package already does this - whew!!)
> Plot track of interest in a diff color as the other detected tracks across frames
> Check to see if there are any sudden jumps. 
> Plot the tracks 

Called by 
~~~~~~~~~
* 20_overall_workflow.py

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
import pathlib
import pandas as pd
import skimage 
import natsort
from difflib import SequenceMatcher as SeqMatch

detailed_description =  """Detect objects from binary images and link them to tracks. 
                        Inputs required are a series of BW images (8-bit). 
                        Outputs generated are CSV  files with the *_tracks_first25_frames pattern,
                        where * is the common portion of the image files. The CSV file
                        is output into the current working directory.
                        
                        Tracking is done using multiple object features and also with motion and 
                        visual tracking updates.
                        """
parser = argparse.ArgumentParser(description=detailed_description)

parser.add_argument('-mask_source', type=str, help='Source folder with file pattern for binarised images. e.g.\
                     /masks/K1/K1*.png . Input needs to be glob-able.')
parser.add_argument('-raw_source', type=str, help='Source folder with file pattern for binarised images. e.g.\
                     /masks/K1/K1*.png . Input needs to be glob-able.')
parser.add_argument('-dest', type=str, default='./', help='Destination for all the cleaned camera images. \
                    Defaults to current folder. Folder will be made if it does not exist')
parser.add_argument('-cam_id', type=str, help='Camera ID')
parser.add_argument('-config', type=pathlib.Path, default='./bat_config.json', help='The \
                    config file path to set up the tracking workflow for btrack. Defaults to ./bat_config.json')
parser.add_argument('-maxsearchradius', type=float, default=80, help='The max search radius within which\
                    a search is conducted while particle-linking to tracks in pixels. Defaults to 80.')

args = parser.parse_args()
#%%
cam_id = args.cam_id
relevant_images = natsort.natsorted(glob.glob(args.mask_source))
bat_detection_masks = np.array([skimage.io.imread(each) for each in relevant_images])
width = bat_detection_masks[0,:,:].shape[1]
height = bat_detection_masks[0,:,:].shape[0]
# get common filename patterns
a,b = [os.path.split(each)[-1] for each in [relevant_images[0], relevant_images[-1]]]
common_range = SeqMatch(None, a, b, autojunk=False).find_longest_match(0,len(a),0,len(b))
common_str = a[:common_range.size]

#%%
FEATURES = ('area','axis_major_length','axis_minor_length','orientation',
             'solidity', 'eccentricity')

objects  = btrack.utils.segmentation_to_objects(bat_detection_masks,
                                                properties=FEATURES)    

CONFIG_FILE = args.config

with btrack.BayesianTracker() as tracker:

   # configure the tracker using a config file
    tracker.configure(CONFIG_FILE)
    tracker.max_search_radius = args.maxsearchradius
    tracker.tracking_updates = ["MOTION","VISUAL"]
    tracker.features = FEATURES

    # append the objects to be tracked
    tracker.append(objects)

    # set the tracking volume
    tracker.volume=((0, width), (0, height)) # the X and Y volume to track!!

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
# cleaned_images = np.array(skimage.io.imread_collection(args.raw_source))

# viewer = napari.Viewer()

# viewer.add_image(
#     np.array(cleaned_images), 
#     name="cleaned",
#     opacity=0.9,
# )

# viewer.add_image(bat_detection_masks, 
#                  name='mask',
#                  opacity=0.2)
# # the track data from the tracker
# viewer.add_tracks(
#     data, 
#     properties=properties, 
#     name="Tracks", 
#     blending="translucent",
# )
tracking_data = pd.DataFrame(data, columns=['id','frame','row','col'])
tracking_data.to_csv(f'{common_str}_tracks_first25_frames.csv')