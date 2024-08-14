import json
import sys
import os
import matplotlib.pyplot as plt
import location_converter.location_converter as lc

from pydantic_models import RouteInput
# Run with
# python src\visualise_deliveries.py "src\data\Locations_local_test.json" "src\data\Locations_local_test_route.json"

def plot_graph():
    locations_path = os.path.join("src", "data", sys.argv[1])
    routes_path = os.path.join("src", "data", sys.argv[2])

    lats, longs, locations, routes = reformat(locations_path,routes_path)
    orders = {order.order_id: {'lat': order.lat, 'long': order.lon} for order in locations.orders}

    #print(orders_dict)
    plt.figure(figsize=(10, 6))
    plt.scatter(longs, lats, color='blue', marker='o')
    __add_lines(routes, orders)
    plt.title('Equirectangular Projection Scatter Plot')
    plt.xlabel('Longitude (km)')
    plt.ylabel('Latitude (km)')
    plt.grid(True)
    plt.show()

def reformat(locations_path, routes_path):
    with open(locations_path, 'r') as file:
        locations = json.load(file)
    with open(routes_path, 'r') as file:
        routes = json.load(file)

    locations = RouteInput(**locations)
    lats = [location.lat for location in locations.orders]
    longs = [location.lon for location in locations.orders]
    lats, longs = lc.geographic_to_2d(lats, longs)
    
    return lats, longs, locations, routes

def __add_lines(routes, orders_dict):
    for route in routes:
        for i in range(len(route) - 1):
            start_id = route[i]
            end_id = route[i + 1]

            start_long = orders_dict[start_id]['long']
            start_lat = orders_dict[start_id]['lat']
            end_long = orders_dict[end_id]['long']
            end_lat = orders_dict[end_id]['lat']

            start_lats, start_longs = lc.geographic_to_2d([start_lat], [start_long])
            end_lats, end_longs = lc.geographic_to_2d([end_lat], [end_long])

            plt.annotate('', xy=(end_longs[0], end_lats[0]), xytext=(start_longs[0], start_lats[0]),
                         arrowprops=dict(facecolor='red', shrink=0.15, headlength=7, headwidth=7, width=3))
    
plot_graph()
