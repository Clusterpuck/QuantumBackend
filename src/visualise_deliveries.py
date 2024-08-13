import json
import matplotlib.pyplot as plt
import location_converter.location_converter as lc

from pydantic_models import RouteInput

def plot_graph(locations_path, routes_path):
    locations, routes = reformat(locations_path,routes_path)
    lats, longs = lc.locations_to_2D(locations.orders)

    orders_dict = {order.order_id: {'lat': order.lat, 'long': order.long} for order in locations.orders}

    print(orders_dict)
    plt.figure(figsize=(10, 6))
    plt.scatter(longs, lats, color='blue', marker='o')

    for route in routes:
        for i in range(len(route) - 1):
            start_id = route[i]
            end_id = route[i + 1]

            start_long = orders_dict[start_id]['long']
            start_lat = orders_dict[start_id]['lat']
            end_long = orders_dict[end_id]['long']
            end_lat = orders_dict[end_id]['lat']

            start_lats, start_longs = lc.geographic_to_2D([start_lat], [start_long])
            end_lats, end_longs = lc.geographic_to_2D([end_lat], [end_long])

            plt.annotate('', xy=(end_longs[0], end_lats[0]), xytext=(start_longs[0], start_lats[0]),
                         arrowprops=dict(facecolor='red', shrink=0.15, headlength=7, headwidth=7, width=3))

    plt.title('Equirectangular Projection Scatter Plot')
    plt.xlabel('Longitude (km)')
    plt.ylabel('Latitude (km)')
    plt.grid(True)
    plt.show()

def reformat(locations_path, routes_path):
    with open(locations_path, 'r', encoding="utf-8") as file:
        locations = json.load(file)
    locations = RouteInput(**locations)

    with open(routes_path, 'r', encoding="utf-8") as file:
        routes = json.load(file)
    return locations, routes
    
locations = 'src/data/Locations_local_test.json'
routes = 'src/data/Locations_local_test_route.json'

plot_graph(locations, routes)