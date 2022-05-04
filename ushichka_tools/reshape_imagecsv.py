# -*- coding: utf-8 -*-
"""
Check correct dimension
=======================
ThermoViewer v2.1.6 has a bug where it exports an extra column such that
the loaded data has 512x641 shape. 

The last column is filled with NaNs when everything is okay. If so, then 
this module deletes the last column and szves a new file.
"""
import argparse
import numpy as np 
import pandas as pd
import os

parser = argparse.ArgumentParser(description='Cleans horizontal and vertical fixed pattern noise')
parser.add_argument('--image', help='Path to image file - only csv file!!')
parser.add_argument('--sep', default=',', help='Which separator to use to parse csv file. Defaults to ;')
parser.add_argument('--output_path', default='', help='The folder where the cleaned image will be saved.')
args = parser.parse_args()

image_path = args.image
output_path = args.output_path
separator = args.sep

# load file
folder, filename = os.path.split(image_path)
only_filename, file_format = filename.split('.')
if file_format == 'csv':
    image_data = pd.read_csv(image_path, header=None, sep=separator).to_numpy()
else:
    raise ValueError(f'Non csv file {image_path} not permitted')

output_path_w_file = os.path.join(output_path,only_filename+'_512x640shape.'+file_format)


# check that the dimensions are 512x640 
if not image_data.shape == (512,640):
    # check that the last column is only nans
    last_column = image_data[:,-1]
    lastcol_isnan = np.all(np.isnan(last_column))    
    if lastcol_isnan:
        correct_df = image_data[:,:-1]
        # check that correct df has the proper shape
        if correct_df.shape==(512,640):
            pd.DataFrame(correct_df).to_csv(output_path_w_file, header=None, index=None)
else:
    raise ValueError(f'{image_path} is already 512x640 shape! No need to run this tool!')