"""
FastAPI framework to provide endpoint to generate routes for Vehicle 
Routing Problem.
Intended to be called only by C# Backend.
"""

import os
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse

from vehicle_clusterer_factory import VehicleClustererFactory
from distance_factory import DistanceFactory
from pydantic_models import Message, RouteInput, Order, OrderInput
from route_optimisation.clusterer.k_means_clusterer import KMeansClusterer
from route_optimisation.recursive_cfrs import RecursiveCFRS
from solver_factory import SolverFactory


app = FastAPI()

# Simple factories
# This relocates and bundles specialised validation/config logic
vehicle_clusterer_factory = VehicleClustererFactory()
distance_factory = DistanceFactory()
solver_factory = SolverFactory()

STATIC_TOKEN = os.environ.get('BACKEND_TOKEN')

def token_authentication(authorisation: str = Header(None)) -> None:
    """
    Authenticates the token of incoming request.

    Parameters
    ----------
    authorisation : str
        'authorisation' header value from HTTP request.

    Raises
    ------
    HTTPException
        Invalid or missing token.
    """
    print("Our Token", STATIC_TOKEN)
    print("Received Token", authorisation)
    if authorisation != f"Bearer {STATIC_TOKEN}":
        raise HTTPException(
            status_code=401, detail="Invalid or missing token",
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


def display_cluster_tree(deep_list: list, depth: int) -> None:
    """
    Prints the content of the cluster tree recursively

    Parameters
    ----------
    deep_list : list
        Contains child lists of customers
    depth : int
        Depth of the cluster tree
    """
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
def default_test() -> None:
    """
    Default page
    """
    return "Switching to FastAPI"

@app.post("/generate-routes", responses={400: {"model": Message}})
async def generate_routes(request: RouteInput,
                          token: str = Depends(token_authentication)) -> list[list[int]]:
    """
    Endpoint provides a solution for the vehicle routing problem (VRP).
    Receives VRP configuration parameters, including a list of orders.

    Parameters
    ----------
    request : RouteInput
        vehicle_cluster_config : ClusterConfig
        solver_config : SolverConfig
        Orders : list[OrderInput]
    token : str
        Verifies incoming token

    Returns
    -------
    output : list[list[int]]
        Outer list contains routes for every vehicle
        Inner list contains route for a particular vehicle
    """
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
    subclusterer = KMeansClusterer(request.solver_config.max_solve_size)

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

    return output
