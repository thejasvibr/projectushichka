# -*- coding: utf-8 -*-
"""
2D object detection correction and Napari utils
===============================================

Created on Mon Aug 21 09:51:30 2023

@author: theja
"""
import numpy as np 
import pandas as pd 
from scipy.interpolate import interp1d
from scipy.spatial import distance



#%% Convert DataFrame to napari track-format
def conv_to_napari_track(df):
    track_data = []
    for i, row in df.iterrows():
        # get only the 
        track_data.append(row.loc['id':].to_numpy())
    return track_data



def replace_or_add_row(main_df, new_data):
    '''
    Input is a pd.DataFrame row. If the row doesn't exist - it is added, otherwise
    all data apart from frame and id numbers can be altered in place.
    
    Parameters
    ----------
    main_df, new_data: pd.DataFrame
        Must have the following columns: row, col, id, frame

    Returns
    -------
    main_df : pd.DataFrame
        The same input df object, but with replaced data or 
        additional rows added in
   
    '''
    for ind, row_data in new_data.iterrows():
        row, col, particle_id, frame = [row_data[each] for each in ['row', 'col', 'id', 'frame']]
        rows_exist = np.logical_and(main_df['id']==particle_id,
                                           main_df['frame']==frame)
        if sum(rows_exist)>0:
            main_df.loc[rows_exist, 'row'] = row
            main_df.loc[rows_exist, 'col'] = col
        else:
            # now create the new rows
            new_row = main_df.index.max()+1
            main_df.loc[new_row,'id'] = particle_id
            main_df.loc[new_row,'frame'] = frame
            main_df.loc[new_row,'row'] = row
            main_df.loc[new_row,'col'] = col
    return main_df




def interpolate_xyz(subdf):
    '''
    Returns an interpolated xyz DataFrame and replaces missing spots 
    with a quadratic spline. 
    
    Parameters
    ----------
    subdf : pd.DataFrame
        With columns id, frame, x, y, z
    
    Returns 
    -------
    subdf_interp : pd.DataFrame
        When there are no missing values a copy of the input subdf is returned.
        When there are missing values a version of subdf with missing cells
        interpolated is returned.
    '''
    nona_subdf = subdf.dropna()
    if nona_subdf.shape[0]==0:
        return nona_subdf
    
    minframe = nona_subdf['frame'].min()
    maxframe = nona_subdf['frame'].max()
    min_to_max = set(np.arange(minframe, maxframe+1))
    current_frames = set(nona_subdf['frame'])
    missing_frames = min_to_max - current_frames
    if len(missing_frames) == 0:
        return subdf.copy()
    
    interpolated_fns = {}
    for axis in ['x','y','z']:
        interpolated_fns[axis] = interp1d(nona_subdf['frame'], nona_subdf[axis], kind=2)
    # now add new rows
    subdf_interp = nona_subdf.copy().reset_index(drop=True)
    for missed_frame in missing_frames:
        last_index_plus1 = subdf_interp.index.max() + 1
        subdf_interp.loc[last_index_plus1,'frame'] = missed_frame
        for axis in ['x','y','z']:
            subdf_interp.loc[last_index_plus1, axis] = interpolated_fns[axis](missed_frame)
    subdf_interp['id'] = subdf['id'].unique()[0]
    return subdf_interp.sort_values('frame')
    

def calc_speed(xyz, fps=25):
    nrows = xyz.shape[0]
    deltatime = 1/fps
    speed_profile = []
    for i in range(1,nrows):
        try:
            travelled_dist = distance.euclidean(xyz[i,:], xyz[i-1,:])
            speed = travelled_dist/deltatime
        except:
            speed = np.nan
        speed_profile.append(speed)
    return speed_profile
