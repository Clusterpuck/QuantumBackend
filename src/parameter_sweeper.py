import ast
import json
import numpy as np

from extended_recursive_cfrs import ExtendedRecursiveCFRS
from route_optimisation.clusterer.k_means_clusterer import KMeansClusterer
from route_optimisation.route_solver.brute_force_solver import BruteForceSolver
from route_optimisation.route_solver.dwave_solver import DWaveSolver
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

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
    orders = get_payload(sys.argv[1])
    tuning_sets = get_tuning_parameters(sys.argv[2])
    solver_parameters = get_solver_parameters(sys.argv[3])
    if len(sys.argv) == 6:
        print("Have argv[5]")
        # TODO Read Exclusion List
        # Have a function for this, use argv[5]

    # Write to file, Check if we can even write in the first place
    with open(os.path.join("data", sys.argv[4]), 'w') as file:
        file.write("Parameter testing test1")

    # Making necessary classes before route generation
    vehicle_clusterer = KMeansClusterer(1)
    subclusterer = KMeansClusterer(solver_parameters.get("max_solve_size"))
    distance_finder = distance_factory.create("cartesian")

    # TODO Setup brute force so we have a baseline
    brute_solver = BruteForceSolver()

    # Hardcoded brute force search
    baseline = ExtendedRecursiveCFRS(vehicle_clusterer,
                                     subclusterer,
                                     distance_finder,
                                     brute_solver,
                                     solver_parameters.get("max_solve_size"))
    
    x, y, base_cost = baseline.solve_vrp(orders)

    # Uses tuning_params list to iterate over
    for set in tuning_sets:
        #solver = create_solver(set)
        vrp_solver = ExtendedRecursiveCFRS(vehicle_clusterer,subclusterer,distance_finder,brute_solver,solver_parameters.get("max_solve_size"))
        vehicle_routes, raw_tree, cost = vrp_solver.solve_vrp(orders)
            # NOTE test with brute force, compare with hardcoded custom route before we go quantum
            # Store whatever our performance metrics are (distance relative to BFS)
        # TODO Append analysis data to file
    # TODO Save matplotlib heatmap to data folder
    # Read the output file for this?
    # TODO Save a contour plot if have time
    # Read the output file for this?
    # TODO Save something to judge individual hyperparams
    # Read the output file for this?
    """print("FROM BASELINE")
    print(x)
    display_cluster_tree(y, 0)
    print(cost)"""


def create_solver(set) -> DWaveSolver:
    base_cost = 10 # For the sake of consistency in testing, please don't change this
    scale_factor = set[0]
    chain_value = set[1]
    return DWaveSolver(
                EmbeddingComposite(
                    DWaveSampler(
                        token=os.environ["DWAVE_TOKEN"],
                        endpoint=os.environ["DWAVE_URL"],
                        solver=os.environ["DWAVE_SOLVER"],
                    )
                ),
                cost_factor = base_cost,
                constraint_factor = base_cost * scale_factor,
                chain_strength = chain_value
            )

# Read payload
def get_payload(file_path : str) -> list[Order]:
    with open(os.path.join("data", file_path), 'r') as file:
        data = RouteInput(**json.load(file))
        data = orders_to_cartesian(data.orders)
        return data
    
def display_cluster_tree(deep_list: list, depth: int) -> None:
    # Assumes correctly formatted cluster tree
    if len(deep_list) != 0:
        if isinstance(deep_list[0], list):
            # Keep searching for list[items] level
            print("  " * depth + f"Split: {depth}")
            for inner_list in deep_list:
                display_cluster_tree(inner_list, depth + 1)
        else:
            # Prints indented cluster leaf
            print("  " * depth + f"Leaf: {[o.order_id for o in deep_list]}")
        # Ignore empty branches (though that could be a bug if so)
    
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
