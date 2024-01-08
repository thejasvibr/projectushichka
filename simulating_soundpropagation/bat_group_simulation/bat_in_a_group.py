# -*- coding: utf-8 -*-
"""
Simulating what a bat hears in a group
======================================
First create objects in a room 

Created on Mon Jan  8 23:38:19 2024

@author: theja
"""
import pyroomacoustics as pra
import numpy as np 
np.random.seed(78464)
import pyvista as pv
import scipy.signal as signal 
import soundfile as sf
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

#%% First set up the positions of the bats in the group. Here we'll just use
# a grid, and then add some noise. 

x_range = np.arange(-1, 1.3, 0.3)
y_range = np.arange(-1, 1.3, 0.3)
xyposns = np.array(np.meshgrid(x_range, y_range)).flatten().reshape(2,-1).T
xyposns += np.random.choice(np.linspace(0,0.1,10), xyposns.size).reshape(-1,2)

plt.figure()
plt.plot(xyposns[:,0], xyposns[:,1],'*')
plt.gca().set_aspect('equal')

xyz_posns = np.column_stack((xyposns, np.tile(5, xyposns.shape[0])))


#%% 
# Place a sphere of a fixed radius at the xyz positions in a large room

plotter = pv.Plotter()

box_room = pv.Box(bounds=[-5, 5, -5, 5, 0, 10], quads=False) # xminmax, yminmax, zminmax
plotter.add_mesh(box_room, opacity=0.1)

spheres = [pv.Sphere(radius=0.1, center=xyz, theta_resolution=10, phi_resolution=10) for xyz in xyz_posns]
for each in spheres:
    plotter.add_mesh(each, color='r')

# now triangulate and save each part as a mesh
box_room.extract_geometry().triangulate().save('bigroom.stl')
# and now the spheres
for i,each in enumerate(spheres):
    each.extract_geometry().triangulate().save(f'sphere_{i}.stl')

plotter.show()
#%%
from stl import mesh
material = pra.Material(energy_absorption=0.2, scattering=0.1)

# with numpy-stl
the_mesh = mesh.Mesh.from_file('bigroom.stl')
ntriang, nvec, npts = the_mesh.vectors.shape
size_reduc_factor = 1  # to get a realistic room size (not 3km)

# create one wall per triangle
walls = []
for w in range(ntriang):
    walls.append(
        pra.wall_factory(
            the_mesh.vectors[w].T / size_reduc_factor,
            material.energy_absorption["coeffs"],
            material.scattering["coeffs"],
        )
    )

# now add one sphere too - and append it to the walls 
for i in range(20):
    sphere_meshes = mesh.Mesh.from_file(f'sphere_{i}.stl')
    ntriang, nvec, npts = sphere_meshes.vectors.shape
    
    for w in range(ntriang):
        walls.append(
            pra.wall_factory(
                sphere_meshes.vectors[w].T / size_reduc_factor,
                material.energy_absorption["coeffs"],
                material.scattering["coeffs"],
            )
        )

#%%
fs = int(500e3)
room = pra.Room(
            walls,
            fs=fs,
            max_order=1,
            ray_tracing=False,
            air_absorption=True,
        )

call_durn = 1e-3
t_call = np.linspace(0, call_durn, int(fs*call_durn))
chirp = signal.chirp(t_call, f0=120e3, f1=30e3, t1=t_call[-1])
chirp *= signal.windows.tukey(chirp.size, alpha=0.1)
chirp *= 0.75

#
# Now let's emit from the origin of the sphere
sphere_1_point = spheres[1].points[0,:] + np.random.normal(0,2e-3,3)
sphere_2_point = np.float64(spheres[1].points[1,:].T )
mic_array  = np.column_stack((sphere_1_point.T, sphere_2_point.T))
room.add_source(np.float64(sphere_1_point), signal=chirp)
room.add_microphone_array(mic_array)

room.image_source_model()
room.ray_tracing()
room.compute_rir()
room.plot_rir()
room.simulate()
# show the room
audio = room.mic_array.signals.T
#%%
plt.figure()
plt.specgram(audio[:,1], Fs=fs, NFFT=128, noverlap=64)
#room.plot(img_order=1)
#%%
plt.figure()
axes  = plt.subplot(111, projection='3d')

# Load the STL files and add the vectors to the plot
your_mesh = [mesh.Mesh.from_file(f'sphere_{i}.stl') for i in range(50)]
for each in your_mesh:
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(each.vectors))
    
    # Auto scale to the mesh size
    scale = each.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
plt.show()