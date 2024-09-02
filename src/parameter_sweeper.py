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
# TODO Fix this RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
# TODO Graph best found route
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
from visualise_deliveries import create_graph

vehicle_clusterer_factory = VehicleClustererFactory()
distance_factory = DistanceFactory()
solver_factory = SolverFactory()

# Helper functions
# TODO Grab this from some other file
def main():
    orders_file = sys.argv[1]
    tuning_file = sys.argv[2]
    solver_file = sys.argv[3]
    output_file = sys.argv[4]

    orders = get_payload(orders_file)
    tuning_sets = get_tuning_parameters(tuning_file)
    solver_parameters = get_solver_parameters(solver_file)
    
    process(orders, tuning_sets, solver_parameters, output_file)

    post_process()
    # Create avg file, heatmap, contour plot
    # Create best file, heatmap, contour plot
    # Create heatmap of failed occurences


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

def process(orders, tuning_sets, solver_parameters, output_file):
    
    matrix = CartesianDistanceFinder().build_matrix(orders)
    brute_solver = BruteForceSolver()
    optimal_route, optimal_cost = brute_solver.solve(matrix)
    print(f"Optimal Route: {optimal_route} costs {optimal_cost}")

    # Log the contents of the file inputs
    write_parameters(orders, tuning_sets, solver_parameters)

    for tuning_set in tuning_sets:
        # Quantum
        #solver = create_solver(tuning_set, solver_parameters)
        for trial in range(1, 6): # 1-5
            relative_cost = 0
            cost = 0
            try:
                route, cost = brute_solver.solve(matrix)
                # Quantum
                #route, cost = solver.solve(matrix)
            except RuntimeError:
                route = []
                cost = -1
                relative_cost = -1
            else:
                relative_cost = cost - optimal_cost

            print(tuning_set[0], tuning_set[1], relative_cost, route)
            trial_df = pd.DataFrame({
                'cost_constraint_ratio': [tuning_set[0]],
                'chain_strength': [tuning_set[1]],
                'relative_cost': [relative_cost],
                'cost': [cost],
                'trial': [trial],
                'route': [route]
             })
            file_exists = os.path.isfile(os.path.join('data', output_file + ".csv"))
            trial_df.to_csv(os.path.join('data', output_file + ".csv"), index=False, mode='a', header=not file_exists)
    print("COMPLETED QUANTUM ROUTES")

def post_process():
    pass
# Call with args, output will be in data folder
# TODO This needs to be reworked, looks terrible
# TODO attach argv's to variables rather than directly calling
"""
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

    # Log the contents of the file inputs
    write_parameters(orders, tuning_sets, solver_parameters)
    
    results = []
    best_results = []
    # Uses tuning_params list to iterate over
    for tuning_set in tuning_sets:
        total_relative_cost = 0
        total_succeeds = 0
        best_cost = 0
        best_route = []
        best_trial = -1
        #solver = create_solver(tuning_set, solver_parameters)
        for trial in range(1, 6): # 1,2,3
            relative_cost = 0
            cost = 0
            try:
                route, cost = brute_solver.solve(matrix)
                #route, cost = solver.solve(matrix)
            except RuntimeError:
                route = []
                cost = -1
                relative_cost = -1 # NOTE Repeated line
            else:
                total_succeeds += 1
                # On first success OR better cost, update best
                if total_succeeds == 1 or cost < best_cost:
                    best_cost = cost
                    best_trial = trial
                    best_route = route
                relative_cost = cost - optimal_cost
                total_relative_cost += relative_cost

            print(tuning_set[0], tuning_set[1], relative_cost, route)
            trial_df = pd.DataFrame({
                'cost_constraint_ratio': [tuning_set[0]],
                'chain_strength': [tuning_set[1]],
                'relative_cost': [relative_cost],
                'cost': [cost],
                'trial': [trial],
                'route': [route]
             })
            file_exists = os.path.isfile(os.path.join('data', sys.argv[4] + ".csv"))
            trial_df.to_csv(os.path.join('data', sys.argv[4] + ".csv"), index=False, mode='a', header=not file_exists)
            # TODO Above 3 Lines in a function
            filename = str(tuning_set[0]) + "_" + str(tuning_set[1]) + "_" + str(trial) + "_" + str(relative_cost)
            create_graph(sys.argv[1], route, "sweep_routes", filename)

        if total_succeeds == 0:
            total_succeeds = 1 # Avoid divide by 0
            avg_cost = best_cost = -1
            r_cost = -1
        else:
            avg_cost = total_relative_cost/total_succeeds
            r_cost = best_cost-optimal_cost
        df = pd.DataFrame({
            'cost_constraint_ratio': [tuning_set[0]],
            'chain_strength': [tuning_set[1]],
            'relative_cost': [avg_cost]
        })
        best_df = pd.DataFrame({
            'cost_constraint_ratio': [tuning_set[0]],
            'chain_strength': [tuning_set[1]],
            'relative_cost': [r_cost],
            'cost': [best_cost],
            'trial': [best_trial],
            'best_route': [best_route]
        })
        results.append(df)
        best_results.append(best_df)

    results_df = pd.concat(results, ignore_index=True)
    best_df = pd.concat(best_results, ignore_index=True)

    results_df.to_csv(os.path.join('data', sys.argv[4] + "avg.csv"), index=False)
    best_df.to_csv(os.path.join('data', sys.argv[4] + "best.csv"), index=False)
    best_df = best_df.drop(['cost', 'trial', 'best_route'], axis=1) # Graphing purposes

    create_heatmap(results_df, "avg")
    create_heatmap(best_df, "best")
    create_contour_plot(results_df, "avg")
    create_contour_plot(best_df, "best")
"""
# TODO add params for index, columns, values, outputfile    
def create_heatmap(results_df : pd.DataFrame, name : str) -> None:
    results_df = results_df.pivot(index='cost_constraint_ratio', columns='chain_strength', values='relative_cost')

    fig = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(results_df, annot=True, cmap='coolwarm_r')
    ax.invert_yaxis()
    plt.title('Heatmap')
    plt.xlabel('chain_strength')
    plt.ylabel('cost_constraint_ratio')
    plt.savefig(os.path.join('data', sys.argv[4] + '_heatmap_' + name), dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_heatmap2(results_df : pd.DataFrame, name : str) -> None:
    results_df = results_df.pivot(index='cost_constraint_ratio', columns='chain_strength', values='failed_routes_count')

    fig = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(results_df, annot=True, cmap='coolwarm_r')
    ax.invert_yaxis()
    plt.title('Heatmap')
    plt.xlabel('chain_strength')
    plt.ylabel('cost_constraint_ratio')
    plt.savefig(os.path.join('data', sys.argv[4] + '_heatmap_' + name), dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_contour_plot(results_df : pd.DataFrame, name : str) -> None:
    results_df = results_df.pivot(index='cost_constraint_ratio', columns='chain_strength', values='relative_cost')

    X = results_df.columns.values
    Y = results_df.index.values
    X, Y = np.meshgrid(X, Y) 
    Z = results_df.values

    fig = plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=10, cmap='coolwarm_r')
    plt.colorbar(contour)
    plt.title('Seaborn Contour Plot Example')
    plt.xlabel('chain_strength')
    plt.ylabel('cost_constraint_ratio')
    plt.savefig(os.path.join('data', sys.argv[4] + '_contours_' + name), dpi=300, bbox_inches='tight')
    plt.close(fig)


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

def write_parameters(orders, tuning_sets, solver_parameters):
    with open(os.path.join("data", sys.argv[4] + "_parameters.txt"), 'w') as file:
        for order in orders:
            file.write(str(order[0]) + '\n')
        file.write(str(tuning_sets) + '\n')
        file.write(str(solver_parameters) + '\n')

def read_csv():
    df = pd.read_csv(os.path.join("data","sweep2_p1.csv"))
    filtered_df = df[df['relative_cost'] != -1]
    print(filtered_df)
    aggregated_df = filtered_df.groupby(['cost_constraint_ratio', 'chain_strength']).agg({
        'relative_cost': 'mean',
        'cost': 'mean'
    }).reset_index()
    print(aggregated_df)
    avg_df = aggregated_df.drop(['cost'], axis=1)
    create_heatmap(avg_df, "temp")

    idx = filtered_df.groupby(['cost_constraint_ratio', 'chain_strength'])['relative_cost'].idxmin()
    min_relative_cost_df = filtered_df.loc[idx].reset_index(drop=True)
    create_heatmap(min_relative_cost_df, "temp2")

    failed_routes_df = df[df['relative_cost'] == -1]
    failed_routes_count = failed_routes_df.groupby(['cost_constraint_ratio', 'chain_strength']).size().reset_index(name='failed_routes_count')
    print(failed_routes_count)
    create_heatmap2(failed_routes_count, "temp3")

#read_csv()
main()