"""Transform latitude and longitude using equirectangular projection"""
import numpy as np

def equi_rect_project(latitudes: list, longitudes: list) -> tuple[np.array, np.array]:
    """
    Use an approximation of equirectangular projection to map latitude and longitude
    to a 2D plane.

    Parameters
    ----------
    latitudes : list of floats
        List of latitudes for orders
    longitudes : list of floats
        List of latitudes for orders

    Returns
    -------
    latitudes : ndarray
        1D, contains array of projected latitudes
    longitudes : ndarray
        1D, contains array of projected longitudes
    """
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)

    r = 6371  # Radius of Earth
    centre_point_deg = -31.952258602714696

    # Convert to radians
    center_latitude_radians = np.radians(centre_point_deg)
    longitudes_radians = np.radians(longitudes)
    latitudes_radians = np.radians(latitudes)

    # Apply equirectangular projection
    longitudes = r * longitudes_radians * np.cos(center_latitude_radians)
    latitudes = r * latitudes_radians

    return latitudes, longitudes
