"""Grid search D-Wave solver parameters for tuning"""

import itertools
import os
import sys
import ast
import json
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

from route_optimisation.route_solver.brute_force_solver import BruteForceSolver
from route_optimisation.route_solver.dwave_solver import DWaveSolver
from route_optimisation.distance_matrix.cartesian_distance_finder import (
    CartesianDistanceFinder,
)
from distance_factory import DistanceFactory
from pydantic_models import RouteInput, Order, OrderInput
from solver_factory import SolverFactory
from vehicle_clusterer_factory import VehicleClustererFactory
from visualise_deliveries import create_graph

# python parameter_sweeper.py "Locations.json" "tuning_params" "solver_params" "output"
# TODO Graph best found route

vehicle_clusterer_factory = VehicleClustererFactory()
distance_factory = DistanceFactory()
solver_factory = SolverFactory()


def main() -> None:
    orders_file = sys.argv[1]
    tuning_file = sys.argv[2]
    solver_file = sys.argv[3]
    output_file = sys.argv[4]

    orders = get_payload(orders_file)
    tuning_sets = get_tuning_parameters(tuning_file)
    solver_parameters = get_solver_parameters(solver_file)

    process(orders, tuning_sets, solver_parameters, output_file)
    post_process(output_file, orders_file)


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
        # solver = create_solver(tuning_set, solver_parameters)
        for trial in range(1, 6):  # 1-5
            relative_cost = 0
            cost = 0
            try:
                route, cost = brute_solver.solve(matrix)
                # Quantum
                # route, cost = solver.solve(matrix)
            except RuntimeError:
                route = []
                cost = -1
                relative_cost = -1
            else:
                relative_cost = cost - optimal_cost

            print(tuning_set[0], tuning_set[1], relative_cost, route)
            trial_df = pd.DataFrame(
                {
                    "cost_constraint_ratio": [tuning_set[0]],
                    "chain_strength": [tuning_set[1]],
                    "relative_cost": [relative_cost],
                    "cost": [cost],
                    "trial": [trial],
                    "route": [route],
                }
            )
            file_exists = os.path.isfile(
                os.path.join("data", output_file + ".csv")
            )
            trial_df.to_csv(
                os.path.join("data", output_file + ".csv"),
                index=False,
                mode="a",
                header=not file_exists,
            )
    print("COMPLETED QUANTUM ROUTES")


def post_process(output_file, orders_file):
    df = pd.read_csv(os.path.join("data", output_file + ".csv"))
    filtered_df = df[df["relative_cost"] != -1]

    average_df = (
        filtered_df.groupby(["cost_constraint_ratio", "chain_strength"])
        .agg({"relative_cost": "mean"})
        .reset_index()
    )
    save_heatmap(
        average_df,
        "avg",
        "cost_constraint_ratio",
        "chain_strength",
        "relative_cost",
        output_file,
    )
    save_contour_plot(
        average_df,
        "avg",
        "cost_constraint_ratio",
        "chain_strength",
        "relative_cost",
        output_file,
    )
    average_df.to_csv(
        os.path.join("data", output_file + "avg.csv"), index=False
    )

    idx = filtered_df.groupby(["cost_constraint_ratio", "chain_strength"])[
        "relative_cost"
    ].idxmin()
    min_relative_cost_df = filtered_df.loc[idx].reset_index(drop=True)
    save_heatmap(
        min_relative_cost_df,
        "best",
        "cost_constraint_ratio",
        "chain_strength",
        "relative_cost",
        output_file,
    )
    save_contour_plot(
        min_relative_cost_df,
        "best",
        "cost_constraint_ratio",
        "chain_strength",
        "relative_cost",
        output_file,
    )
    min_relative_cost_df.to_csv(
        os.path.join("data", output_file + "best.csv"), index=False
    )

    failed_routes_df = df[df["relative_cost"] == -1]
    if not failed_routes_df.empty:
        failed_routes_count = (
            failed_routes_df.groupby(
                ["cost_constraint_ratio", "chain_strength"]
            )
            .size()
            .reset_index(name="failed_routes_count")
        )
        save_heatmap(
            failed_routes_count,
            "fails",
            "cost_constraint_ratio",
            "chain_strength",
            "failed_routes_count",
            output_file,
        )

    # visualise deliveries
    for _, row in df.iterrows():
        cost_constraint_ratio = str(row["cost_constraint_ratio"])
        chain_strength = str(row["chain_strength"])
        trial = str(row["trial"])
        relative_cost = str(row["relative_cost"])
        filename = (
            f"{cost_constraint_ratio}_{chain_strength}_{trial}_{relative_cost}"
        )
        route_list = ast.literal_eval(row["route"])
        create_graph(orders_file, route_list, "sweep_routes", filename)


def save_heatmap(
    results_df: pd.DataFrame,
    name: str,
    index: str,
    columns: str,
    values: str,
    outputfile: str,
) -> None:
    print(results_df)
    results_df = results_df.pivot(index=index, columns=columns, values=values)

    fig = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(results_df, annot=True, cmap="coolwarm_r")
    ax.invert_yaxis()
    plt.title("Heatmap")
    plt.xlabel("chain_strength")
    plt.ylabel("cost_constraint_ratio")
    plt.savefig(
        os.path.join("data", outputfile + "_heatmap_" + name),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_contour_plot(
    results_df: pd.DataFrame,
    name: str,
    index: str,
    columns: str,
    values: str,
    outputfile: str,
):
    results_df = results_df.pivot(index=index, columns=columns, values=values)

    X = results_df.columns.values
    Y = results_df.index.values
    X, Y = np.meshgrid(X, Y)
    Z = results_df.values

    fig = plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=10, cmap="coolwarm_r")
    plt.colorbar(contour)
    plt.title("Contour Plot")
    plt.xlabel("chain_strength")
    plt.ylabel("cost_constraint_ratio")
    plt.savefig(
        os.path.join("data", outputfile + "_contours_" + name),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_solver(set, solver_params: dict) -> DWaveSolver:
    cost_factor = solver_params.get("cost_factor")
    num_runs = solver_params.get("num_runs")
    max_retries = solver_params.get("max_retries")
    is_circuit = solver_params.get("is_circuit")
    scale_factor = set[0]
    chain_value = set[1]
    constraint_factor = cost_factor * scale_factor
    print("CREATING: ", scale_factor, chain_value)
    return DWaveSolver(
        EmbeddingComposite(
            DWaveSampler(
                token=os.environ["DWAVE_TOKEN"],
                endpoint=os.environ["DWAVE_URL"],
                solver=os.environ["DWAVE_SOLVER"],
            )
        ),
        cost_factor=cost_factor,
        constraint_factor=constraint_factor,
        num_runs=num_runs,
        max_retries=max_retries,
        is_circuit=is_circuit,
        chain_strength=chain_value,
    )


# Read payload
def get_payload(file_path: str) -> list[tuple[Order, Order]]:
    with open(os.path.join("data", file_path), "r", encoding="utf-8") as file:
        data = RouteInput(**json.load(file))
        orders = orders_to_cartesian(data.orders)
        orders = [(o, o) for o in orders]
        if len(orders) > 9:
            raise ValueError("Potentially too many orders being iterated")
        return orders


# Read parameters
def get_tuning_parameters(file_path: str) -> list[tuple[int, int]]:
    result = {}
    with open(os.path.join("data", file_path), "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = ast.literal_eval(value.strip())
                result[key] = value
    result = list(
        itertools.product(
            result.get("cost_constraint_ratio"), result.get("chain_strength")
        )
    )
    return result


# Read D-Wave settings
def get_solver_parameters(file_path: str) -> dict:
    parameters = {}
    with open(os.path.join("data", file_path), "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = ast.literal_eval(
                    value.strip()
                )  # Convert value to the appropriate type
                parameters[key] = value
    return parameters


def write_parameters(
    orders: list[tuple[Order, Order]],
    tuning_sets: list[tuple[int, int]],
    solver_parameters: dict,
) -> None:
    with open(
        os.path.join("data", sys.argv[4] + "_parameters.txt"),
        "w",
        encoding="utf-8",
    ) as file:
        for order in orders:
            file.write(str(order[0]) + "\n")
        file.write(str(tuning_sets) + "\n")
        file.write(str(solver_parameters) + "\n")


main()
