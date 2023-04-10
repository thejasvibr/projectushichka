# -*- coding: utf-8 -*-
"""
An overall workflow to create bat 3D tracks
===========================================
Created on Fri Apr  7 13:10:24 2023

@author: theja
"""
import pathlib
import subprocess

#%% 
# Generate bat detection binary masks and the 2D locations of all the detected
# objects just in case they are needed.

common_command = "python 0_generate_bat_masks.py -dest cleaned_and_masks   -start 0 -end 50 "
raw_images = {camera : pathlib.Path(f'2018-08-17/{camera}/P001/png/*.png') for camera in ['K1', 'K2', 'K3']}
cam_specific = {camera : f"-source {raw_images[camera]} -cam_id {camera}" for camera in ['K1', 'K2', 'K3']}
print('K1 data running...')
subprocess.call(common_command + cam_specific['K1'])
print('K2 data running...')
subprocess.call(common_command + cam_specific['K2'])
print('K3 data running...')
subprocess.call(common_command + cam_specific['K3'])

#%%
# Generate tracks from the 2D masks. the `btrack` package does object detection
# and linking itself - but output needs to be corrected every now and then. 

common_link_command = "python 1_link_to_tracks.py"
for cam in ['K1','K2','K3']:
    print(f'Running {cam} object detection and tracking')
    cam_specific1 = f" -mask_source .\cleaned_and_masks\masks\{cam}\mask_{cam}*.png"
    cam_specific2 = f" -raw_source .\cleaned_and_masks\cleaned\{cam}\cleaned_{cam}_*.png -dest ./ -cam_id {cam}"
    whole_command = common_link_command + cam_specific1 + cam_specific2
    subprocess.call(whole_command)

# Now proceed to verify the tracking using the interactive Napari viewer, and 
# implement the required corrections. 


#%% 
# Get 2D tracks and correct them
# Run traj_correction_2018-08-17_P01-7000TMC-first25f.py and inspect the tracking + implement corrections. 

import traj_correction_2018_08_17_P01_7000TMC_first25f

#%%
# Manually match the 2D tracks across cameras
subprocess.call('python 3_manual_trajmatching_2018-_08_17_7000TMC-first25f.py')


