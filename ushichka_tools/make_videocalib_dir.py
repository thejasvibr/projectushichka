# -*- coding: utf-8 -*-
"""
Template video-calibration folder structure
===========================================
Creates empty directories that will hold video calibration data. The
folder structure is as following

YYYY-MM-DD
    - image
    - mics
    - other_cave_surface
    - video_calibration
        - axis
        - calibration_output
        - wand

Created on Thu May  5 11:24:17 2022

@author: thejasvi
"""
import argparse
import os


parser = argparse.ArgumentParser(description='Creates empty directories that will hold video calibration data.')
parser.add_argument('--parent',default='.', help='The parent folder inside which the sub-directories will\
                    be created')
args = parser.parse_args()

parent_folder = args.parent

pathjoin = os.path.join
def ifnotexist_make(fullpath):
    if not os.path.exists(fullpath):
        os.mkdir(fullpath)
    else:
        raise ValueError(f'{fullpath} already exists - cannot make directory')

if __name__ == '__main__':    
    outer_folder_names = ['image', 'mics', 'other_cave_surface', 'video_calibration']
    for each in outer_folder_names:
        ifnotexist_make(pathjoin(parent_folder,each))
    
    # Now the video calibration folder and its subfolders
    video_calib_path = pathjoin(parent_folder,'video_calibration')
    # now make video calib subfolders
    subfolders = ['axis', 'calibration_output', pathjoin('calibration_output','round1'),'gravity', 'wand']
    for eachsub in subfolders:
        ifnotexist_make(pathjoin(video_calib_path, eachsub))
