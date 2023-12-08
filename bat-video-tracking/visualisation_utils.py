# -*- coding: utf-8 -*-
"""
Module to generate visualisations of bat trajectories in PyVista
================================================================

Created on Sat Sep  9 17:10:58 2023

@author: theja
"""
import numpy as np
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt




def generate_incave_video(mesh, traj_data, cam_props, output_vidname, fps=25, **kwargs):
    '''

    Parameters
    ----------
    mesh : pyvista PolyData
        Cave mesh
    traj_data : pd.DataFrame
        With columns frame, id, x, y, z
    camera_props : dict
        Properties of the camera. A dictionary with the following keys
        position (3, np.array), azimuth (deg), elevation (deg), roll (deg),
        view_angle (deg)
    output_vidname : str
        Name of the output MP4 video WITHOUT the .mp4 extension.
    fps : int>0, optional
        Frame rate at which the data was captured. The default is 25.
    colormap : str, optional
        Defaults to 'viridis'.
    radius : float>0, optional
        Radius of sphere used to represent bat trajectories.
        Defaults to 0.15 m
    writegif : bool, True
        Whether to write a gif or mp4 file. 
        Defaults to True.

    Returns
    -------
    None. Generates a .mp4 file with ```output_vidname```

    Notes
    -----
    Based heavily ont he PyVista example page here
    https://docs.pyvista.org/version/stable/examples/02-plot/movie.html
    '''
    if kwargs.get('writegif', True):
        output_video = output_vidname + '.gif'
    else:
        output_video = output_vidname + '.mp4'
        
    
    unique_pointids = np.unique(traj_data['id'])
    byid = traj_data.groupby('id')
    valid_pointids = []
    for each in unique_pointids:
        subdf = byid.get_group(each)
        if np.all(np.isnan(subdf.loc[:,'x':'z'].to_numpy())):
            pass
        else:
            valid_pointids.append(each)
    
    cmap = plt.get_cmap(kwargs.get('colormap', 'viridis'))
    colors = [cmap(i) for i in np.linspace(0,1,len(valid_pointids))]
    id2colormap = {batid : idcolor for batid, idcolor in zip(valid_pointids, colors)}
    
    plotter = pv.Plotter()
    if kwargs.get('writegif', True):
        plotter.open_gif(output_video, fps=fps)
    else:
        plotter.open_movie(output_video, framerate=fps)

    
    
    plotter.camera.position = cam_props['position']
    plotter.camera.azimuth = cam_props['azimuth']
    plotter.camera.roll = cam_props['roll']
    plotter.camera.elevation = cam_props['elevation']
    
    plotter.camera.view_angle = cam_props['view_angle']
        
    # add cave mesh
    plotter.add_mesh(mesh, opacity=0.3)
    
    plotter.write_frame()  # write initial data
    
    trajdata_sorted = traj_data.copy().sort_values('frame')
    
    # add 'guide-lines'
    for batid, subdf in trajdata_sorted.groupby('id'):
        if batid in valid_pointids:
            plotter.add_lines(subdf.loc[:,'x':'z'].to_numpy(),
                              color=id2colormap[batid],
                              connected=True, width=1)
    
    for frame, subdf in trajdata_sorted.groupby('frame'):
        thisframe_points = {}
        for batid, batdf in subdf.groupby('id'):
            if batid in valid_pointids:
                thisframe_points[batid] = pv.Sphere(radius=kwargs.get('radius', 0.15),
                                                  center=batdf[['x','y','z']].to_numpy(),
                                                  )
        actors = []
        for batid, batmesh in thisframe_points.items():
            actor = plotter.add_mesh(batmesh, color=id2colormap[batid])
            actors.append(actor)
        plotter.add_text(f"Time: {frame/fps}", name='time-label')
        plotter.write_frame()  # Write this frame
        # Remove all bat meshes
        for actor in actors:
            plotter.remove_actor(actor)

    plotter.close()