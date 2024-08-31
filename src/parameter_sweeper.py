import ast
import json
import numpy as np

"""
Optimal parameters are unknown, we must incorporate a method to sweep for optimal D-Wave parameters. Add everything to a separate folder. Its goal must be able to provide sufficient data to allow parameter sets to be analysed.

Input = sets of D-Wave hyperparams, these will have to be added to the solver factory.
Input = Exclusion list, exclude a particular combination, rather have this earlier than later
Output = Output file storing data for analysis

Structure

- Preprocessing + relevant data to begin processing
- Loop over the combination of parameters
  - Call generate routes function with D-Wave solver of a configuration
  - Write results to file

Will need some metric for "optimality". Could compare costs of BFS and Quantum?
2 parameters, can do simple graph?
Heatmap?
Contour plot?
"""
# python .\parameter_sweeper.py "newFile"
# Plan
import itertools
import os
import sys
from distance_factory import DistanceFactory
from pydantic_models import RouteInput, Order, OrderInput
from solver_factory import SolverFactory
from vehicle_clusterer_factory import VehicleClustererFactory

vehicle_clusterer_factory = VehicleClustererFactory()
distance_factory = DistanceFactory()
solver_factory = SolverFactory()
# Args should be
    # File path for payload
    # File path for params
    # File path for output

# Helper functions
# TODO Grab this from some other file
def orders_to_cartesian(
    orders: list[OrderInput],
) -> list[Order]:
    """
    Pre-compute Cartesians for all orders.

    Lat/long's distortion is not ideal for estimating distances in clustering
    and solving, so this is used in place of real traffic distances etc.

    Parameters
    ----------
    orders: list of OrderInput
        The raw structs containing Customer IDs, latitude and longitude

    Returns
    -------
    list of Order
        Orders with additional x, y, z coordinate info
    """
    # NOTE: Doesn't cohere well with routing code, being an input pre-processor

    r = 6371  # Radius of the earth
    cartesian_orders = []

    for order in orders:
        r_lat, r_lon = np.deg2rad(order.lat), np.deg2rad(order.lon)
        cartesian_orders.append(
            Order(
                order_id=order.order_id,
                lat=order.lat,
                lon=order.lon,
                x=r * np.cos(r_lat) * np.cos(r_lon),
                y=r * np.cos(r_lat) * np.sin(r_lon),
                z=r * np.sin(r_lat),
            )
        )

    return cartesian_orders

# Call with args, output will be in data folder
def wrapper():
    # Read payload, must be quantum
    # Read tuning_parameters, get combinations of params
    # Read solver_params
    # Write to file, Check if it can write before we start
    # Make the clusterers, distance finder via factories
    # Setup brute force so we have a baseline
    # for each combination of params
        # Recreate the solver with new params
        # solve_vrp, extend and override so I can get cost
            # test with brute force, compare with hardcoded custom route before we go quantum
        # Append analysis data to file
    # Save matplotlib heatmap to data folder
    # Save a contour plot if have time
    # Save something to judge individual hyperparams
    f = open(os.path.join("data", sys.argv[1]), "x")
    f.write("Now the file has more content")
    f.close()


# Read payload
def read_payload(file_path : str) -> list[Order]:
    order_list = []
    with open(os.path.join("data", file_path), 'r') as file:
        data = RouteInput(**json.load(file))
        data = orders_to_cartesian(data.orders)
        return data
    
    # Read payload file, get orders
    # Use orders_to_cartesian before returning

# Read parameters
def get_tuning_parameters(file_path : str) -> list:
    # Plan
    # Read file to get cost-constraint ratio and chain
    result = {}
    with open(os.path.join("data", file_path), 'r') as file:
        for line in file:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = ast.literal_eval(value.strip())
                result[key] = value
    print(type(result.get("cost_constraint_ratio")))
    result = get_combinations(result.get("cost_constraint_ratio"), result.get("chain_strength"))
    print(result)
    return result

# TODO Can just combine this with above
def get_combinations(list1, list2):
    list3 = list(itertools.product(list1, list2))
    #print(list3)
    return list3

# Read D-Wave settings
def get_solver_parameters(file_path : str) -> dict:
    parameters = {}
    with open(os.path.join("data", file_path), 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = ast.literal_eval(value.strip())  # Convert value to the appropriate type
                print(value)
                print(type(value))
                parameters[key] = value
    # TODO Validate Params before return
    print(parameters)
    return parameters

get_solver_parameters("solver_params")
get_tuning_parameters("tuning_params")
print(read_payload("Locations.json"))
