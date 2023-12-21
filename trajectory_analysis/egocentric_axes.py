# -*- coding: utf-8 -*-
"""
Calculating egocentric coordinate systems and ray-tracing
=========================================================



Created on Sun Dec 10 23:38:14 2023

@author: theja
"""
from itertools import product
import numpy as np 
from scipy.spatial import transform, distance_matrix


def get_azimuth(xyz):
    x,y,z = xyz
    return np.arctan2(y, x)
def get_elevation(xyz):
    x,y,z = xyz
    return np.arctan2(z,np.sqrt(x**2+y**2))

def get_xyz(r,theta,phi):
    '''
    The R, theta (elevation), phi (azimuth)
    
    References
    ----------
    Wikipedia page 'Spherical coordinate system', 'Cartesian coordinates' sub-heading
    accessed 10/12/2023
    
    '''
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return np.array([x,y,z])

def calculate_2egocentric_xyz(flight_direction, roll):
    '''
    Calculates the local xyz vectors with roll.
    
    Parameters
    ----------
    flight_direction : (3,) np.array
        Unit vector representing the flight direction. 

    Returns 
    -------
    with_roll_xyz : (3,3) np.array
        Each row represents the local x,y,z vectors
    
    '''
    
    # get the perpendicular of the y and anti-gravity vector 
    x_the_perpendicular = np.cross(flight_direction, [0,0,1])
    z_the_perpendicular = np.cross(x_the_perpendicular, flight_direction )
    # the local xyz without any roll
    almost_egocentric = np.row_stack((x_the_perpendicular,
                                      flight_direction,
                                      z_the_perpendicular))

    # now calculate the rotation in the direction of flight 
    rotation_about_flight = transform.Rotation.from_rotvec(flight_direction*roll)
    with_roll_xyz = rotation_about_flight.apply(almost_egocentric)

    return with_roll_xyz

def make_rays(phi_vals=np.linspace(0,360,9),
                      theta_vals=np.linspace(0,180,11)):
    '''
    Makes rays of unit length emanating from the global origin 0,0,0
    
    TODO
    ----
    * Detect redundant theta-phi combinations?

    Parameters
    ----------
    phi_vals : (N,) np.array 
        The azimuth angles in degrees. Defaults to 5 rays spread out around 360 deg. 
    theta_vals : (M,) np.array
        The elevation angles in degrees. Defaults to 10 rays spread out around 180 deg. 

    Returns 
    -------
    unique_phi_theta : (N,2) np.array
        Azimuth and elevation values in the 0th and 1st column respectively.
    unique_rays : (N,3) np.array
        Unit vectors that correspond to the rays centred on the origin.
    
    '''
    theta_phi_combis = list(product(phi_vals.tolist(), theta_vals.tolist()))
   
    all_ray_vectos = []
    for (phi, theta) in theta_phi_combis:
        xyz = get_xyz(1, np.radians(theta), np.radians(phi))
        all_ray_vectos.append(xyz)
    
    all_rays = np.array(all_ray_vectos)
    all_rays[np.abs(all_rays)<1e-15] = 0
    # # simplify as all_rays has repeats 
    # unique_rays, row_inds = np.unique(all_rays, axis=0, return_index=True)
    # unique_phi_theta = [theta_phi_combis[each] for each in row_inds]
    return theta_phi_combis, all_rays



def find_ray_intersection(cave, xyz, ray_vector, amp_factor=50):
    '''
    Finds the closest point to the ray_vector. If none is found then 
    a np.nan is returned. 

    Parameters
    ----------
    cave : pv.PolyData
        Mesh
    xyz : (3,) np.array
    ray_vector: (3,) np.array
        Unit vector with the ray direction 
    amp_factor : float>1
        Amount by which to multiply the ray_vector
    
    Returns 
    -------
    points :np.array
        Points in the cave that coincide with the exteneded ray
    '''
    extended_ray = xyz + ray_vector*amp_factor
    points, ind = cave.ray_trace(xyz, extended_ray)
    if points.size>3:
        all_dists = distance_matrix(xyz.reshape(1,3), points)
        closest_point_ind = np.argmin(all_dists)
        points = points[closest_point_ind,:]
    elif points.size==0:
        points = np.tile(np.nan,3)
    return points, extended_ray

def get_all_ray_intersections(rays, *args):
    cave, xyz, amp_factor = args
    all_intersection_points = np.zeros(rays.shape)
    all_extended_rays = np.zeros(rays.shape)
    for row in range(rays.shape[0]):
        point, extended_ray = find_ray_intersection(cave, xyz, rays[row,:])
        all_intersection_points[row,:] = point
        all_extended_rays[row,:] = extended_ray
    all_intersection_points = np.array(all_intersection_points).reshape(-1,3)
    return all_intersection_points, all_extended_rays
    