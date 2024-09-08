"""
Driver code to visualise routing data for testing purposes

Arguments
---------
1. The name of the JSON file in src/data containing RouteInput data
2. the name of the JSON file in src/data containing the optimal routes per vehicle

Example:

When in src directory (cd src), assuming you have a src/data folder containing all the objects, call
- python3 visualise_deliveries.py "Locations.json" "Locations_route.json"
"""

import json
import sys
import os
import matplotlib.pyplot as plt
import route_visualiser.er_projection as erp

from pydantic_models import RouteInput   

def create_graph(locations_file, route, folder_name, file_name) -> None:
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

    lats, longs, orders = reformat_locations(locations_path)

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(longs, lats, color='blue', marker='o')
    __add_lines(route, orders)
    plt.title('Equirectangular Projection Scatter Plot')
    plt.xlabel('Longitude (km)')
    plt.ylabel('Latitude (km)')
    plt.grid(True)
    plt.savefig(os.path.join('data', folder_name, file_name + ".png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def reformat_locations(locations_path):
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
    lats, longs = erp.equi_rect_project(lats, longs)
    orders = {order.order_id: {'lat': order.lat, 'lon': order.lon} for order in locations.orders}

    return lats, longs, orders

def __add_lines(routes, orders_dict):
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

    start_lats, start_longs = erp.equi_rect_project(start_lat, start_longs)
    end_lats, end_longs = erp.equi_rect_project(end_lats, end_longs)

    for i in range(len(start_lats)):
        plt.annotate('', xy=(end_longs[i], end_lats[i]), xytext=(start_longs[i], start_lats[i]),
                 arrowprops=dict(facecolor='red', shrink=0.15, headlength=7, headwidth=7, width=3))

#plot_graph()
#temp_list = [[3, 0, 1, 2, 5, 6, 4]]
#create_graph("Locations.json", temp_list, "testFolder", "testFile")
