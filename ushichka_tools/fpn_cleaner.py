# -*- coding: utf-8 -*-
"""
Fixed pattern noise cleaning 
============================
Accepts image and csv file formats
"""

import argparse
import imageio
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

get_file_format = lambda X: os.path.split(X)[-1]

if __name__ == '__main__':
    # load file
    folder, filename = os.path.split(image_path)
    only_filename, file_format = filename.split('.')
    if file_format == 'csv':
        image_data = pd.read_csv(image_path, header=None, sep=separator).to_numpy()
    else:
        raise ValueError(f'Non csv file {image_path} not permitted')
    
    cleaned_image = remove_vert_horiz_lines(image_data)
    # save file to output path
    output_path_w_file = os.path.join(output_path,only_filename+'_FPNcleaned.'+file_format)
    output_path_image = os.path.join(output_path,only_filename+'_FPNcleaned.png')
    if file_format == 'csv':
        pd.DataFrame(cleaned_image).to_csv(output_path_w_file, header=None, index_label=None)
        to_int_image = cleaned_image.copy()
        to_int_image /= np.max(to_int_image)
        to_int_image *= 255
        to_int_image = np.uint8(to_int_image)
        imageio.imwrite(output_path_image, to_int_image)    