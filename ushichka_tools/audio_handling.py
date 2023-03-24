# -*- coding: utf-8 -*-
"""
Ushichka Audio processing tools
===============================
Utility functions to perform simple and routine multichannel audio-video stuff. 

Created on Fri Mar 24 14:12:53 2023

@author: theja

"""
import numpy as np 
import scipy.ndimage as ndi


def first_frame_index(sync_channel, thresh=0.1):
    '''
    Parameters
    ----------
    sync_channel : (,1) np.array
        25 Hz Square signal 
    Returns
    -------
    first_frame : int
        Index of the sync_channel at which the first frame fired. 
    '''
    sync = sync_channel.copy()
    sync -= np.mean(sync)
    sync /= np.max(np.abs(sync))
    frame_fires = sync > thresh
    
    labels, num_objs = ndi.label(frame_fires)
    if num_objs < 1 :
        raise ValueError('< 1 frame detected -- check input sync audio!')
    frame_inds = ndi.find_objects(labels)
    first_frame = frame_inds[0][0].start
    return first_frame