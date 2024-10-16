import random
from fastapi.responses import JSONResponse
import numpy as np
import os
import math
from fastapi import FastAPI, HTTPException, Depends, Header
import time
import httpx
import threading

from vehicle_clusterer_factory import VehicleClustererFactory
from distance_factory import DistanceFactory
from pydantic_models import Message, RouteInput, Order, OrderInput, DepotInput, Depot
from route_optimisation.clusterer.k_means_clusterer import KMeansClusterer
from route_optimisation.recursive_cfrs import RecursiveCFRS
from solver_factory import SolverFactory

def task():
    for i in range(5):
        print(f"Task is running: {i}")
        time.sleep(5)  # Simulate some work being done

def main_task():
    pass

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

def depot_reorder(route: list[Order], depot: DepotInput) -> list[int]:
    """
    Reorder a route to minimise the extra cost of considering depot location

    Parameters
    ----------
    route: list of Order
        Ordered route containing orders
    depot: DepotInput
        position of depot

    Returns
    -------
    new_route: list of int
        Ordered list of order IDs after considering depot position
    """
    # Convert to cartesian
    depot = depot_to_cartesian(depot)
    
    best_length = -float('inf')
    best_pair = None
    for i in range(len(route) - 1, -1, -1):
        start = route[i]
        end = route[(i + 1) % len(route)]

        # Maximise start_to_end, minimise everything between depots
        start_to_end = math.dist((start.x, start.y, start.z), (end.x, end.y, end.z))
        depot_to_start = math.dist((depot.x, depot.y, depot.z), (start.x, start.y, start.z))
        end_to_depot = math.dist((end.x, end.y, end.z), (depot.x, depot.y, depot.z))
        length = start_to_end - (depot_to_start + end_to_depot)
        if length > best_length:
            best_length = length
            best_pair = i # Index of route's end
    
    # Reorder routes according to new start and end indices
    start_index = best_pair + 1
    end_index = best_pair
    if start_index == len(route)-1:
        reordered_route = route[end_index+1:] + route[:start_index]
    else:
        reordered_route = route[start_index:] + route[:end_index + 1]

    # Recreate route with only order ids
    new_route = [order.order_id for order in reordered_route]

    return new_route

# Endpoints
@app.get("/")
def default_test():
    return "Switching to FastAPI"

async def main_task(request):
    time.sleep(260)
    # try:
    #     vehicle_clusterer = vehicle_clusterer_factory.create(
    #         request.vehicle_cluster_config
    #     )
    #     distance_finder = distance_factory.create(request.solver_config.distance)
    #     route_solver = solver_factory.create(request.solver_config.type)
    # except ValueError as e:
    #     # Should be safe to relay these back to client
    #     return JSONResponse(status_code=400, content={"message": str(e)})

    # # For recursive, we need to cap max clusters, since it stitches on return
    # # Since capturing substructures matters progressively less, just k-means it
    # subclusterer = KMeansClusterer(
    #     request.solver_config.max_solve_size,
    #     allow_less_data=True,
    #     duplicate_clusters="split",
    # )

    # vrp_solver = RecursiveCFRS(
    #     vehicle_clusterer,
    #     subclusterer,
    #     distance_finder,
    #     route_solver,
    #     request.solver_config.max_solve_size,
    # )

    # # Pre-compute Cartesian approx, since it's very likely we will use it
    # new_orders = orders_to_cartesian(request.orders)

    # try:
    #     # Solve VRP
    #     optimal_route_per_vehicle, cluster_tree = vrp_solver.solve_vrp(new_orders)
    # except ValueError as e:
    #     return JSONResponse(
    #         status_code=400,
    #         content={"message": f"Validation error. Payload args may be invalid: {e}"},
    #     )
    # except RuntimeError as e:
    #     return JSONResponse(
    #         status_code=400,
    #         content={
    #             "message": f"Error at runtime. Payload data or args may be invalid: {e}"
    #         },
    #     )

    # # Print clustering results to console
    # display_cluster_tree(cluster_tree, 0)

    # # Extract just the IDs, keeping double nested shape
    # output = [[o.order_id for o in v] for v in optimal_route_per_vehicle]

    # # Reorder according to depot
    # if request.depot is not None:
    #     for i, route in enumerate(optimal_route_per_vehicle):
    #         if len(route) > 1: # Don't do anything for single order routes
    #             output[i] = depot_reorder(route, request.depot)
    output = [[1, 5, 542]]
    
    async with httpx.AsyncClient() as client:
        response = await client.post("https://routingdata.azurewebsites.net/api/Calculation", json=output)
        if response.status_code != 200:
            print("Failed to notify:", response.text)

@app.post("/generate-routes", responses={400: {"model": Message}})
async def generate_routes(
    request: RouteInput, token: str = Depends(token_authentication)
):
    # Create Task
    print(request)
    thread = threading.Thread(target=main_task, args=(request,))
    print("Before Thread")
    thread.start()
    print("After Thread")
    # Call function
    # return
    print(request.model_dump())
    return 1
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
        for i, route in enumerate(optimal_route_per_vehicle):
            if len(route) > 1: # Don't do anything for single order routes
                output[i] = depot_reorder(route, request.depot)

    # async with httpx.AsyncClient() as client:
    #     response = await client.post("https://example.com/api/notify", json=output)
    #     if response.status_code != 200:
    #         print("Failed to notify:", response.text)
    
    # RUN THE TASK

    #return output
