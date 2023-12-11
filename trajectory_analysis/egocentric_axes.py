# -*- coding: utf-8 -*-
"""
Calculating egocentric coordinate systems
=========================================



Created on Sun Dec 10 23:38:14 2023

@author: theja
"""

import numpy as np 
from scipy.spatial import transform

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


def calculate_egocentric_xyz(flight_direction, roll, ):
    '''
    
    Parameters
    ----------
    flight_direction : (3,) np.array
        Unit vector with the flight direction from t_N-1 to t_N
    
    roll : float
        Roll in radians. Here 0 roll is when the vector points to the 
        north pole. 

    Returns
    -------
    egocentric_xyz : (3,3) np.array
        The rotated xyz coordinates, centred on the origin itself.
        Each row is one axis' vector.
    
    Note
    ----
    Translation to the actual trajectory needs to be done separately. 
    
    '''
    # This is azimuth and elevation of the flight-aligned y-axis
    azimuth, elevation = get_azimuth(flight_direction), get_elevation(flight_direction)
    
    # now calculate the azimuth of the x-axis
    azimuth_x = azimuth - np.pi/2
    # and the roll is the elevation. (top is 0)
    elevation_x = roll
    calculated_xaxis = get_xyz(1, elevation_x, azimuth_x, )
    # The z-axis is the upward pointing vector that is perpendicular to 
    # the x and y.
    calculated_zaxis = np.cross(calculated_xaxis, flight_direction)
    egocentric_xyz = np.row_stack((calculated_xaxis, 
                                   flight_direction, 
                                   calculated_zaxis))
    return egocentric_xyz


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


# #%%
# if __name__ == "__main__":
#     pp = pv.Plotter()
    
#     pp.add_mesh(pv.Arrow(direction=x_the_perpendicular), color='r')
#     pp.add_mesh(pv.Arrow(direction=flight_direction), color='g')
#     pp.add_mesh(pv.Arrow(direction=z_the_perpendicular), color='b')
#     pp.show()


