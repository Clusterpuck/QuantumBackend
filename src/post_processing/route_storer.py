"""Saves routes to data folder"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_dir)

from pydantic_models import RouteInput

def create_graph(locations_file: str, route: list[int], folder_name: str, file_name: str) -> None:
    """
    Creates a graph and saves to a folder

    Parameters
    ----------
    locations_path : str
        The name of the locations file at src/data
        The file should be in the format of RouteInput
    route : list[int]
        Route of index ids
    folder_name : str
        Name of the folder to be saved to
    file_name : str
        Name of the file to be saved
    """
    locations_path = os.path.join("data", locations_file)
    route = [route]

    lats, longs, orders = __reformat_locations(locations_path)

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(longs, lats, color='blue', marker='o')
    __add_lines(route, orders)
    plt.title('Equirectangular Projection Scatter Plot')
    plt.xlabel('Longitude (km)')
    plt.ylabel('Latitude (km)')
    plt.grid(True)
    plt.savefig(os.path.join('data', folder_name, file_name + ".png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def __reformat_locations(locations_path: str) -> tuple[np.array, np.array, dict[int, dict[str,float]]]:
    """
    Extracts the latitudes, longitudes and orders from JSON Object

    Parameters
    ----------
    locations_path : str
        The name of the locations file at src/data
        The file should be in the format of RouteInput

    Returns
    -------
    lats : np.array
        List of latitudes for orders 
    longs : np.array
        List of latitudes for orders
    orders : dict[int, dict[str,float]]
        Key represents order_id
        Value (inner dictionary) is in the format of:
            lat: float
            lon: float
        Example: {16: {'lat': -31.899364, 'lon': 115.801288}
    """
    with open(locations_path, 'r', encoding='utf-8') as file:
        locations = json.load(file)

    locations = RouteInput(**locations)
    lats = [location.lat for location in locations.orders]
    longs = [location.lon for location in locations.orders]
    lats, longs = __equi_rect_project(lats, longs)
    orders = {order.order_id: {'lat': order.lat, 'lon': order.lon} for order in locations.orders}

    return lats, longs, orders

def __add_lines(routes: list[list[int]], orders_dict: dict[int, dict[str,float]]) -> None:
    """
    Adds arrows for each edge in routes list

    Parameters
    ----------
    routes : list[list[int]]
        Contains a list of lists of routes in sorted order
        Outer lists contains the list of routes
        Inner list contains the orders in sorted order
    orders_dict : dict[int, dict[str,float]]
        Key represents order_id
        Value (inner dictionary) is in the format of:
            lat: float
            lon: float
        Example: {16: {'lat': -31.899364, 'lon': 115.801288}
    """
    start_longs = []
    start_lat = []
    end_longs = []
    end_lats = []
    for route in routes:
        for i in range(len(route) - 1):
            start_id = route[i]
            end_id = route[i + 1]

            start_longs.append(orders_dict[start_id]['lon'])
            start_lat.append(orders_dict[start_id]['lat'])
            end_longs.append(orders_dict[end_id]['lon'])
            end_lats.append(orders_dict[end_id]['lat'])

    start_lats, start_longs = __equi_rect_project(start_lat, start_longs)
    end_lats, end_longs = __equi_rect_project(end_lats, end_longs)

    for i in range(len(start_lats)):
        plt.annotate('', xy=(end_longs[i], end_lats[i]), xytext=(start_longs[i], start_lats[i]),
                 arrowprops=dict(facecolor='red', shrink=0.15, headlength=7, headwidth=7, width=3))

def __equi_rect_project(latitudes: list, longitudes: list) -> tuple[np.array, np.array]:
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
    centre_point_deg = -31.952258602714696 # Hardcoded to approximate centre of Perth

    # Convert to radians
    center_latitude_radians = np.radians(centre_point_deg)
    longitudes_radians = np.radians(longitudes)
    latitudes_radians = np.radians(latitudes)

    # Apply equirectangular projection
    longitudes = r * longitudes_radians * np.cos(center_latitude_radians)
    latitudes = r * latitudes_radians

    return latitudes, longitudes
