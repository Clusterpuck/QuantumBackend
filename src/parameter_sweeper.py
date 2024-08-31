import ast
import json
import numpy as np

"""
Optimal parameters are unknown, we must incorporate a method to sweep for optimal D-Wave parameters. Add everything to a separate folder. Its goal must be able to provide sufficient data to allow parameter sets to be analysed.

Input = sets of D-Wave hyperparams, these will have to be added to the solver factory.
Input = Payload
Input = D-Wave other Params, read from file for params that are semi-fixed
Input = Exclusion list, exclude a particular combination, rather have this earlier than later
Output = Output file storing data for analysis, relevant heatmap/contour plot as images

Structure

Preprocessing + relevant data to begin processing
Loop over the combination of parameters
Call generate routes function with D-Wave solver of a configuration
Write results to file
Cost metric will be distance relative to BFS
"""
# python parameter_sweeper.py "Locations.json" "tuning_params" "solver_params"
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
    # Read our 3 inputs
    print(get_payload(sys.argv[1]))
    print(get_tuning_parameters(sys.argv[2]))
    print(get_solver_parameters(sys.argv[3]))
    if len(sys.argv) < 5:
        print("Have argv[4]")
    # TODO Read Exclusion List

    # TODO Write to file, Check if it can write before we start
    # Have a function for this, use argv[4]
    # TODO Make the clusterers, distance finder via factories
    # Will have to use results from file getters
    # TODO Setup brute force so we have a baseline
    # Hardcoded brute force search, assume recursive clustering is impossible
    # TODO for each combination of params
    # Uses tuning_params list to iterate over
        # TODO Recreate the solver with new params
        # Should only be D-Wave solver being resetted
        # TODO solve_vrp, extend and override so I can get cost
            # NOTE test with brute force, compare with hardcoded custom route before we go quantum
            # Store whatever our performance metrics are (distance relative to BFS)
        # TODO Append analysis data to file
    # TODO Save matplotlib heatmap to data folder
    # Read the output file for this?
    # TODO Save a contour plot if have time
    # Read the output file for this?
    # TODO Save something to judge individual hyperparams
    # Read the output file for this?


# Read payload
def get_payload(file_path : str) -> list[Order]:
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
    result = get_combinations(result.get("cost_constraint_ratio"), result.get("chain_strength"))
    return result

# TODO Can just combine this with above
def get_combinations(list1, list2):
    list3 = list(itertools.product(list1, list2))
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
                parameters[key] = value
    # TODO Validate Params before return
    return parameters

wrapper()
"""get_solver_parameters("solver_params")
get_tuning_parameters("tuning_params")
read_payload("Locations.json")"""
