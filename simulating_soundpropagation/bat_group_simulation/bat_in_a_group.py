# -*- coding: utf-8 -*-
"""
Simulating what a bat hears in a group
======================================
First create 'bats' in a room and then let one of them emit a call.


Things to try out
-----------------
* Place two 'ears' that are very close to the source of the call
* Implement call directionality and microphone directionality
* Currently the faint echoes are barely visible - but this is more a visualisation thing
as the 'direct' call is over 6 orders of magnitude louder than anything else. 


TODO
----
* Figure out those pesky normals :|

Created on Mon Jan  8 23:38:19 2024

@author: theja
"""
import pyroomacoustics as pra
import numpy as np 
np.random.seed(78464)
import pyvista as pv
import os 
import glob
import scipy.signal as signal 
import soundfile as sf
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

#%% First set up the positions of the bats in the group. Here we'll just use
# a grid, and then add some noise. 

x_range = np.linspace(-3, 3.3, 10)
y_range = np.linspace(-3, 3.3, 10)
#xyposns = np.array(np.meshgrid(x_range, y_range)).flatten().reshape(2,-1).T
xyposns = np.column_stack((x_range, y_range))
alternate_rows = np.arange(xyposns.shape[0])[::2]
xyposns[alternate_rows,0] += 1.5
#xyposns += np.random.choice(np.linspace(0.8,1.5,10), xyposns.size).reshape(-1,2)
#%%
plt.figure()
plt.plot(xyposns[:,0], xyposns[:,1],'*')
plt.gca().set_aspect('equal')

stl_files = glob.glob('*room*.stl')+glob.glob('sphere*.stl')
if len(stl_files)>0:
    map(os.remove, stl_files)

#subset_inds = 
xyz_posns = np.column_stack((xyposns, np.tile(5, xyposns.shape[0])))


#%% 
# Place a sphere of a fixed radius at the xyz positions in a large room

plotter = pv.Plotter()

box_room = pv.Box(bounds=[-15, 15, -15, 15, 0, 10], quads=False) # xminmax, yminmax, zminmax
plotter.add_mesh(box_room, opacity=0.1)

line_x = pv.Line([-5,0,5],[5,0,5])
line_y = pv.Line([0,-5,5], [0,5,5])
plotter.add_mesh(line_x, color='red')
plotter.add_mesh(line_y, color='green')

spheres = [pv.Sphere(radius=0.3, center=xyz, theta_resolution=40, phi_resolution=40) for xyz in xyz_posns]
for each in spheres:
    plotter.add_mesh(each, color='r')
sphere_centers = []
sphere_labels = []

for i, each in enumerate(spheres):
    sphere_centers.append(each.center)
    sphere_labels.append(f'# {i}')

plotter.add_point_labels(sphere_centers, sphere_labels, point_size=20, font_size=36)


def callback(a, b, distance):
    plotter.add_text(f'Echo delay: {distance*2/343}', name='dist')
plotter.add_measurement_widget(callback)

# now triangulate and save each part as a mesh
box_room.extract_geometry().triangulate().save('bigroom.stl')
# and now the spheres
for i,each in enumerate(spheres):
    each.compute_normals(inplace=True, flip_normals=True)
    each.extract_geometry().triangulate().save(f'sphere_{i}.stl')

#plotter.show(cpos='xy')
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

scenario = ''#'0th sphere + 1'
#now add one sphere too - and append it to the walls 
for i in range(len(spheres)):
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
fs = int(250e3)


#%%
#
# Now let's emit from the origin of the sphere
idxnum = 9

sphere_northpole = spheres[idxnum].points[0,:]
source_xyz = sphere_northpole + np.array([0,0, 1e-2])
#sphere_2_point = np.float64(spheres[idxnum].points[1,:].T )
mic_xyz = sphere_northpole + np.array([[0,2e-3,5e-3],
                                    [0,-2e-3,5e-3]])



# #%%
# plot2 = pv.Plotter()
# #for each in spheres[:2]:
# plot2.add_mesh(spheres[idxnum], )
# #plot2.add_mesh(spheres[1])
# npoints = spheres[idxnum].points.shape[0]
# plot2.add_points(mic_xyz, color='blue')
# plot2.add_points(source_xyz, color='black')
# # #plot2.add_point_labels(spheres[idxnum].points, range(npoints))
# plot2.show()




#%%
print('...initialising room...')
room = pra.Room(
            walls,
            fs=fs,
            max_order=1,
            ray_tracing=False,
            air_absorption=True,
        )

call_durn = 0.5e-3
t_call = np.linspace(0, call_durn, int(fs*call_durn))
chirp = signal.chirp(t_call, f0=90e3, f1=40e3, t1=t_call[-1])
chirp *= signal.windows.tukey(chirp.size, alpha=0.1)
chirp *= 0.75
print('...room initialised...')
room.add_source(np.float64(source_xyz), signal=chirp)
# now add all the other 'bat-calls' 
non_hearing = set(range(len(spheres))) - set([idxnum])
other_sources = np.array([spheres[each].points[0,:] + np.array([0,0,1e-2]) for each in non_hearing])
emission_times = np.random.choice(np.linspace(0,40e-3, 30), other_sources.shape[0])

for othersource, t_emission in zip(other_sources, emission_times):
    room.add_source(np.float64(othersource), delay=t_emission)

room.add_microphone_array(np.float64(mic_xyz.T))

print('.....running...')
room.image_source_model()
room.ray_tracing()
room.compute_rir()
#room.plot_rir()
room.simulate()
# show the room
audio = room.mic_array.signals.T
print('....done...')
#%%
chnum = 0
f, t, sxx = signal.spectrogram(audio[:,chnum], fs=fs, nperseg=96, noverlap=48)
call_times = t<= call_durn
#sxx[:,call_times] = 1e-6

plt.figure()
a0 = plt.subplot(211)
plt.imshow(20*np.log10(sxx), origin='lower', aspect='auto', extent=[t[0], t[-1], 0, f[-1]])
a0.set_title(scenario)
plt.subplot(212, sharex=a0)
log_rir = np.log10(np.abs(room.rir[chnum][0]))
plt.plot(np.linspace(0, log_rir.size/fs, log_rir.size), log_rir)

#room.plot(img_order=1)
#%%
# plt.figure()
# axes  = plt.subplot(111, projection='3d')

# # Load the STL files and add the vectors to the plot
# your_mesh = [mesh.Mesh.from_file(f'sphere_{i}.stl') for i in range(50)]
# for each in your_mesh:
#     axes.add_collection3d(mplot3d.art3d.Poly3DCollection(each.vectors))
    
#     # Auto scale to the mesh size
#     scale = each.points.flatten()
# axes.auto_scale_xyz(scale, scale, scale)

# # Show the plot to the screen
# plt.show()