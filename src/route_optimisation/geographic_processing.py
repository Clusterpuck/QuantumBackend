"""Handles geographic processing of points"""

import numpy as np

from pydantic_models import Order

def geographic_array(delivery_list):
    """
    Grab the longitude and latitude from delivery_list

    Parameters
    ----------
    delivery_list: pandas.DataFrame
        Dataframes containing Customer IDs, latitude and longitude

    Returns
    -------
    lat_long_array: numpy.ndarray
        array of arrays containing latitude and longitude
    """
    lat_long_array = delivery_list[['Latitude', 'Longitude']].to_numpy()
    return lat_long_array

def geographic_to_cartesian(geo_array):
    """
    convert geographic locations to 3D cartesian coordinates

    Parameters
    ----------
    geo_array: numpy.ndarray
        Contains latitude and longitude

    Returns
    -------
    cartesian_array: numpy.ndarray
        Contains x,y,z coordinate on cartesian plane
    """
    cartesian_array = np.zeros((geo_array.shape[0],3))
    for index, x in enumerate(geo_array):
        cartesian_coord = get_cartesian(x[0],x[1])
        cartesian_array[index] = cartesian_coord
    return cartesian_array

def get_cartesian(lat: float, lon: float):
    """
    Get cartesian coordinate for a latitude and longitude point

    Parameters
    ----------
    lat: float
        latitude value
    lon: float
        longitude value

    Returns
    -------
    numpy.ndarray
        Contains x,y,z coordinate on cartesian plane
    """
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    r = 6371 # radius of the earth
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array((x,y,z))

def orders_to_cartesian(orders: list[Order]) -> np.ndarray:
    """
    Convert a delivery list to a cartesian array

    Parameters
    ----------
    delivery_list: pandas.DataFrame
        Dataframes containing Customer IDs, latitude and longitude

    Returns
    -------
    cartesian_array: numpy.ndarray
        Contains x,y,z coordinate on cartesian plane
    """
    geo_array = geographic_array(orders)
    cartesian_array = geographic_to_cartesian(geo_array)
    #print(cartesian_array)
    return cartesian_array
