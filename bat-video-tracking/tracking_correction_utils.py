# -*- coding: utf-8 -*-
"""
2D object detection correction and Napari utils
===============================================

Created on Mon Aug 21 09:51:30 2023

@author: theja
"""
import numpy as np 
import pandas as pd 



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
