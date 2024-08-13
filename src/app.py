import random
from fastapi.responses import JSONResponse
import numpy as np
from fastapi import FastAPI, HTTPException

from distance_factory import DistanceFactory
from pydantic_models import Fact, RouteInput, Order, OrderInput
from route_optimisation.recursive_cfrs import RecursiveCFRS
from solver_factory import SolverFactory


app = FastAPI()

facts = ["One", "Two", "Three", "Four", "Five"]

# Simple factories
# This relocates and bundles specialised validation/config logic
# clustering_factory = ClusteringFactory()
distance_factory = DistanceFactory()
solver_factory = SolverFactory()


# Helper functions
def get_total_facts():
    return len(facts)


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


# Endpoints
@app.get("/")
def default_test():
    return "Switching to FastAPI"


@app.get("/randomfact")
def random_fact():
    total_facts = get_total_facts()
    if total_facts is None:
        # Sample exception handling
        raise HTTPException(status_code=500, detail="Failed to retrieve facts")

    random_fact_id = random.randint(0, total_facts - 1)

    return {"fact": facts[random_fact_id]}


@app.post("/addfact")
def add_fact(new_fact: Fact):
    facts.append(new_fact.fact)
    return {"message": "Fact added successfully", "total_facts": get_total_facts()}


@app.post("/generate-routes")
async def generate_routes(request: RouteInput):
    # Input should already be type/range validated by pydantic

    # Since requests should be stateless and unshared, set up new solvers
    try:
        # vehicle_clustering = clustering_factory.create(request.vehicle_cluster_config)
        # subclustering = clustering_factory.create(request.subcluster_config)
        dm = distance_factory.create(request.solver_config.distance)
        rs = solver_factory.create(request.solver_config.type)
    except ValueError as e:
        # Should be safe to relay these back to client
        return JSONResponse(status_code=400, content={"message": str(e)})
    split_threshold = 3
    vrp_solver = RecursiveCFRS(dm, rs, split_threshold)

    # Pre-compute Cartesian approx, since it's very likely we will use it
    new_orders = orders_to_cartesian(request.orders)

    # Solve VRP
    optimal_route_per_vehicle, cluster_tree = vrp_solver.solve_vrp(
        new_orders, request.vehicle_cluster_config.k
    )
    # TODO: The cluster rework will handle vehicle count k instead, once done

    # Print clustering results to console
    display_cluster_tree(cluster_tree, 0)

    # Extract just the IDs, keeping double nested shape
    output = [[o.order_id for o in v] for v in optimal_route_per_vehicle]

    return output
