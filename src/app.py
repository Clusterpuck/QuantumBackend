import random
from fastapi.responses import JSONResponse
import numpy as np
import os
import math
from fastapi import FastAPI, HTTPException, Depends, Header

from vehicle_clusterer_factory import VehicleClustererFactory
from distance_factory import DistanceFactory
from pydantic_models import Message, RouteInput, Order, OrderInput, DepotInput, Depot
from route_optimisation.clusterer.k_means_clusterer import KMeansClusterer
from route_optimisation.recursive_cfrs import RecursiveCFRS
from solver_factory import SolverFactory


app = FastAPI()

facts = ["One", "Two", "Three", "Four", "Five"]

# Simple factories
# This relocates and bundles specialised validation/config logic
vehicle_clusterer_factory = VehicleClustererFactory()
distance_factory = DistanceFactory()
solver_factory = SolverFactory()

STATIC_TOKEN = os.environ.get("BACKEND_TOKEN")


def token_authentication(authorisation: str = Header(None)):
    # Apparently you should add Bearer?
    print("Our Token", STATIC_TOKEN)
    print("Received Token", authorisation)
    if authorisation != f"Bearer {STATIC_TOKEN}":
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing token",
        )


# Helper functions
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

def depot_to_cartesian(depot_input: DepotInput) -> Depot:
    """
    Convert depot latitude and longitude to 3D Cartesian

    Parameters
    ----------
    depot_input: DepotInput
        contains depot's latitude and longitude

    Returns
    -------
    depot: Depot
        depot with additional x, y, z coordinate info
    """
    r = 6371
    r_lat, r_lon = np.deg2rad(depot_input.lat), np.deg2rad(depot_input.lon)
    depot = Depot(
        lat=depot_input.lat,
        lon=depot_input.lon,
        x=r * np.cos(r_lat) * np.cos(r_lon),
        y=r * np.cos(r_lat) * np.sin(r_lon),
        z=r * np.sin(r_lat),
    )
    return depot


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

def depot_reorder(route: list[int], orders: list[Order], depot: DepotInput) -> list[int]:
    """
    Reorder a route to minimise the extra cost of considering depot location

    Parameters
    ----------
    route: list of int
        Ordered list of order IDs
    orders: list of Order
        list of all orders in VRP
    depot: DepotInput
        position of depot

    Returns
    -------
    new_route: list of int
        Ordered list of order IDs after considering depot position
    """
    # Convert to cartesian
    depot = depot_to_cartesian(depot)

    # Create a dictionary for quick access
    orders_dict = {order.order_id: order for order in orders}
    
    # Generate every list of every pair
    pairs = []
    n = len(route)
    for i in range(n):
        pair = (route[i], route[(i + 1) % n]) # Should wrap around
        pairs.append(pair)

    # Loop to find best pair
    best_length = -float('inf')
    best_pair = None
    for pair in pairs:
        end = orders_dict.get(pair[0])
        start = orders_dict.get(pair[1])

        # Maximise start_to_end, minimise everything between depots
        start_to_end = math.dist((start.x, start.y, start.z), (end.x, end.y, end.z))
        depot_to_start = math.dist((depot.x, depot.y, depot.z), (start.x, start.y, start.z))
        end_to_depot = math.dist((end.x, end.y, end.z), (depot.x, depot.y, depot.z))
        length = start_to_end - (depot_to_start + end_to_depot)
        if length > best_length:
            best_length = length
            best_pair = pair

    # Reconstruct route with best pair
    start_index = route.index(best_pair[1])
    end_index = route.index(best_pair[0])
    if start_index < end_index:
        # start_index is before end
        new_route = route[start_index:end_index + 1]
        new_route.extend(route[:start_index])
        new_route.extend(route[end_index + 1:])
    else:
        # start_index is after end
        new_route = route[start_index:] + route[:end_index + 1]

    return new_route

# Endpoints
@app.get("/")
def default_test():
    return "Switching to FastAPI"


@app.post("/generate-routes", responses={400: {"model": Message}})
async def generate_routes(
    request: RouteInput, token: str = Depends(token_authentication)
):
    # Input should already be type/range validated by pydantic

    # Since requests should be stateless and unshared, set up new solvers
    try:
        vehicle_clusterer = vehicle_clusterer_factory.create(
            request.vehicle_cluster_config
        )
        distance_finder = distance_factory.create(request.solver_config.distance)
        route_solver = solver_factory.create(request.solver_config.type)
    except ValueError as e:
        # Should be safe to relay these back to client
        return JSONResponse(status_code=400, content={"message": str(e)})

    # For recursive, we need to cap max clusters, since it stitches on return
    # Since capturing substructures matters progressively less, just k-means it
    subclusterer = KMeansClusterer(
        request.solver_config.max_solve_size,
        allow_less_data=True,
        duplicate_clusters="split",
    )

    vrp_solver = RecursiveCFRS(
        vehicle_clusterer,
        subclusterer,
        distance_finder,
        route_solver,
        request.solver_config.max_solve_size,
    )

    # Pre-compute Cartesian approx, since it's very likely we will use it
    new_orders = orders_to_cartesian(request.orders)

    try:
        # Solve VRP
        optimal_route_per_vehicle, cluster_tree = vrp_solver.solve_vrp(new_orders)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Validation error. Payload args may be invalid: {e}"},
        )
    except RuntimeError as e:
        return JSONResponse(
            status_code=400,
            content={
                "message": f"Error at runtime. Payload data or args may be invalid: {e}"
            },
        )

    # Print clustering results to console
    display_cluster_tree(cluster_tree, 0)

    # Extract just the IDs, keeping double nested shape
    output = [[o.order_id for o in v] for v in optimal_route_per_vehicle]

    # Reorder according to depot
    if request.depot is not None:
        for i, route in enumerate(output):
            if len(route) > 1: # Don't do anything for single order routes
                output[i] = depot_reorder(route, new_orders, request.depot)

    return output
