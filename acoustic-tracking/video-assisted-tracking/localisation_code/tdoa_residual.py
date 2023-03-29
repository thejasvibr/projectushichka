# -*- coding: utf-8 -*-
"""
TDOA residual calculation
=========================
Taken from the PYDATEMM library

Created on Wed Mar 29 15:31:14 2023

@author: theja
"""
import numpy as np 
import scipy.spatial as spatial 
euclidean = spatial.distance.euclidean


def residual_tdoa_error_nongraph(d, source, array_geom, **kwargs):
    '''
    Parameters
    ----------
    d : (Nchannels-1) np.array
        Range differences
    array_geom : (Nchannels,3) np.array
    kwargs:
        'c' : v of sound in m/s
    '''
    obs_d = d.copy()
    n_channels = array_geom.shape[0]
    source_array_dist = np.apply_along_axis(euclidean, 1, array_geom, source)
    # the TDOA vector measured from data
    obtained_n_tilde = source_array_dist[1:] - source_array_dist[0]
    obtained_n_tilde /= kwargs.get('c', 343.0)
    # tdoa residual 
    obs_d /= kwargs.get('c', 343.0)
    tdoa_resid = euclidean(obs_d, obtained_n_tilde)
    tdoa_resid /= np.sqrt(n_channels)
    return tdoa_resid
