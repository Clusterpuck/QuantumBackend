import ast
import json
import numpy as np

from route_optimisation.route_solver.brute_force_solver import BruteForceSolver
from route_optimisation.route_solver.dwave_solver import DWaveSolver
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from route_optimisation.distance_matrix.cartesian_distance_finder import CartesianDistanceFinder

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
# python parameter_sweeper.py "Locations.json" "tuning_params" "solver_params" "output"
import itertools
import os
import sys
from distance_factory import DistanceFactory
from pydantic_models import RouteInput, Order, OrderInput
from solver_factory import SolverFactory
from vehicle_clusterer_factory import VehicleClustererFactory
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Setting the ground truth optimal route and cost
    dm = CartesianDistanceFinder()
    matrix = dm.build_matrix(orders)
    brute_solver = BruteForceSolver()
    optimal_route, optimal_cost = brute_solver.solve(matrix)
    print(f"Optimal Route: {optimal_route} costs {optimal_cost}")

    with open(os.path.join("data", sys.argv[4] + "_parameters.txt"), 'w') as file:
        for order in orders:
            file.write(str(order[0]) + '\n')
        file.write(str(tuning_sets) + '\n')
        file.write(str(solver_parameters) + '\n')
    
    results = []
    # Uses tuning_params list to iterate over
    for tuning_set in tuning_sets:
        total_relative_cost = 0
        total_succeeds = 0
        for x in range(3):
            #solver = create_solver(tuning_set, solver_parameters)
            route, cost = brute_solver.solve(matrix)
            try:
                route, cost = brute_solver.solve(matrix)
                #route, cost = solver.solve(matrix)
            except RuntimeError:
                route = []
                cost = optimal_cost - 1
                total_succeeds += 1
            else:
                total_succeeds += 1
                relative_cost = cost - optimal_cost
                total_relative_cost += relative_cost    
            print(tuning_set[0], tuning_set[1], relative_cost, route)
            trial_df = df = pd.DataFrame({
                'cost_constraint_ratio': [tuning_set[0]],
                'chain_strength': [tuning_set[1]],
                'relative_cost': [relative_cost],
                'route': [route]
             })
            file_exists = os.path.isfile(os.path.join('data', sys.argv[4] + ".csv"))
            trial_df.to_csv(os.path.join('data', sys.argv[4] + ".csv"), index=False, mode='a', header=not file_exists)

        if total_succeeds == 0:
            total_succeeds = 1
        avg_cost = total_relative_cost/total_succeeds
        df = pd.DataFrame({
            'cost_constraint_ratio': [tuning_set[0]],
            'chain_strength': [tuning_set[1]],
            'relative_cost': [avg_cost]
        })
        results.append(df)
        
    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(os.path.join('data', sys.argv[4] + "avg.csv"), index=False)
    create_heatmap(results_df)
    create_contour_plot(results_df)
    # TODO Save something to judge individual hyperparams?
    
def create_heatmap(results_df) -> None:
    results_df = results_df.pivot(index='cost_constraint_ratio', columns='chain_strength', values='relative_cost')

    plt.figure(figsize=(8, 6))
    sns.heatmap(results_df, annot=True, cmap='coolwarm_r')

    plt.title('Heatmap')
    plt.xlabel('chain_strength')
    plt.ylabel('cost_constraint_ratio')
    plt.savefig(os.path.join('data', sys.argv[4] + '_heatmap'), dpi=300, bbox_inches='tight')

def create_contour_plot(results_df):
    results_df = results_df.pivot(index='cost_constraint_ratio', columns='chain_strength', values='relative_cost')

    X = results_df.columns.values
    Y = results_df.index.values
    X, Y = np.meshgrid(X, Y) 
    Z = results_df.values

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=10, cmap='coolwarm_r')
    plt.colorbar(contour)
    plt.title('Seaborn Contour Plot Example')
    plt.xlabel('chain_strength')
    plt.ylabel('cost_constraint_ratio')
    plt.savefig(os.path.join('data', sys.argv[4] + '_contours'), dpi=300, bbox_inches='tight')


def create_solver(set, solver_params : dict) -> DWaveSolver:
    cost_factor = solver_params.get('cost_factor')
    num_runs = solver_params.get('num_runs')
    max_retries = solver_params.get('max_retries')
    is_circuit = solver_params.get('is_circuit')
    scale_factor = set[0]
    chain_value = set[1]
    constraint_factor = cost_factor * scale_factor
    print("CREATING: ", scale_factor, chain_value)
    #print(cost_factor, num_runs, max_retries, is_circuit, scale_factor, chain_value)
    return DWaveSolver(
                EmbeddingComposite(
                    DWaveSampler(
                        token=os.environ["DWAVE_TOKEN"],
                        endpoint=os.environ["DWAVE_URL"],
                        solver=os.environ["DWAVE_SOLVER"],
                    )
                ),
                cost_factor= cost_factor,
                constraint_factor= constraint_factor,
                num_runs= num_runs,
                max_retries= max_retries,
                is_circuit= is_circuit,
                chain_strength= chain_value
            )

# Read payload
def get_payload(file_path : str) -> list[tuple[Order, Order]]:
    with open(os.path.join("data", file_path), 'r') as file:
        data = RouteInput(**json.load(file))
        orders = orders_to_cartesian(data.orders)
        orders = [(o, o) for o in orders]
        if len(orders) > 9:
            raise ValueError("Potentially too many orders being iterated")
        return orders

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
    # TODO Validate Params before return?
    return parameters

wrapper()