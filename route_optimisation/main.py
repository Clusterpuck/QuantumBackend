import math
import numpy as np
from sklearn.cluster import KMeans

from pydantic_models import CartesianOrder, Order
from distance_matrix.distance_matrix_context import DistanceMatrixContext
from route_solver.route_solver_context import RouteSolverContext


def orders_to_cartesian(
    orders: list[Order],
) -> list[CartesianOrder]:
    """
    Pre-compute cartesians for all orders. Required since lat/long's distortion
    is not ideal for estimating distances in clustering and solving.

    Parameters
    ----------
    orders: list of Order
        List of structs containing Customer IDs, latitude and longitude

    Returns
    -------
    list of CartesianOrder
        Contains additional x, y, z coordinates on a Cartesian plane
    """
    r = 6371  # radius of the earth
    cartesian_orders = []
    for order in orders:
        # Convert to cartesian
        r_lat, r_lon = np.deg2rad(order.lat), np.deg2rad(order.long)
        r = 6371  # radius of the earth

        # Add as extended object
        cartesian_orders.append(
            CartesianOrder(
                order.order_id,
                order.lat,
                order.long,
                r * np.cos(r_lat) * np.cos(r_lon),
                r * np.cos(r_lat) * np.sin(r_lon),
                r * np.sin(r_lat),
            )
        )
    return cartesian_orders


def cartesian_cluster(orders: list[CartesianOrder], k: int):
    points = [[o.x, o.y, o.z] for o in orders]
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(points)
    return kmeans.labels_


def solve_cluster(
    orders: list[CartesianOrder],
    distance_matrix: DistanceMatrixContext,
    route_solver: RouteSolverContext,
):
    """
    Find the optimal route and cost for a node

    Parameters
    ----------
    orders: list of Order
        Unordered list of Customer IDs, latitude and longitude
    distance_matrix: DistanceMatrix
        Strategy to create the distance matrix
    route_solver: RouteSolver
        Strategy to finding the optimal route

    Returns
    -------
    list of Order
        Orders in optimal visit order
    """
    # NOTE: The difference between cartesian solve and traffic is that one can use the precomputed xyz, while the other much fetch via api
    # TODO: Will move new implementation into strategy once approved

    n = len(orders)
    matrix = np.zeros([n] * 2)

    # Fill it the lazy way
    for i in range(n):
        for j in range(n):
            if i != j:
                order_a = orders[i]
                order_b = orders[j]
                matrix[i, j] = math.dist(
                    [order_a.x, order_a.y, order_a.z], [order_b.x, order_b.y, order_b.z]
                )

    route_solver.solve(matrix)


def subdivide_routes(
    orders: list[CartesianOrder],
    split_threshold: int,
    dm: DistanceMatrixContext,
    rs: RouteSolverContext,
) -> tuple[list[CartesianOrder], list]:
    # Split oversized clusters further
    labels = cartesian_cluster(orders, split_threshold)
    # Aligned cluster labels

    # Try solving each cluster
    optimal_route = []
    tree_reconstruction = []
    for cluster in np.unique(labels):
        order_indicies = np.where(labels == cluster)[0]
        if order_indicies.size > split_threshold:
            # Too big, subdivide further
            subset_orders = [orders[i] for i in order_indicies]
            subdivide_routes(subset_orders, split_threshold)
        else:
            # Safe size to solve
            solved_child, collected_shape = solve_cluster(orders, dm, rs)
            optimal_route.extend(solved_child)

            # Since it's useful for debugging, build the tree in reverse
            tree_reconstruction.append(collected_shape)

    return optimal_route, tree_reconstruction


def partition_vehicles(
    orders: list[Order],
    vehicles: int,
    split_threshold: int,
    dm: DistanceMatrixContext,
    rs: RouteSolverContext,
) -> tuple[list[list[CartesianOrder]], list]:
    # The main entry wrapper

    # Validate params
    # The presence of pydantic means only range checking is required
    seen_id = set()
    for order in orders:
        if order in seen_id:
            raise ValueError("order_id must be unique")
        elif not (-90 <= order.lat <= 90):
            raise ValueError("lat must be between -90 and 90")
        elif not (-180 <= order.long <= 180):
            raise ValueError("long must be between -180 and 180")
        
        seen_id.add(order)

    # Pre-compute Cartesians
    orders = orders_to_cartesian(orders)

    # Preliminary cluster into distinct vehicle routes
    labels = cartesian_cluster(orders, vehicles)

    # TODO: This duplication can be compressed further
    # Try solving each cluster
    optimal_route = []
    tree_reconstruction = []
    for cluster in np.unique(labels):
        order_indicies = np.where(labels == cluster)[0]
        if order_indicies.size > split_threshold:
            # Too big, subdivide further
            subset_orders = [orders[i] for i in order_indicies]
            subdivide_routes(subset_orders, split_threshold)
        else:
            # Safe size to solve
            solved_child, collected_shape = solve_cluster(orders, dm, rs)
            optimal_route.extend(solved_child)

            # Since it's useful for debugging, build the tree in reverse
            tree_reconstruction.append(collected_shape)

    return optimal_route, tree_reconstruction


# TODO: Reuse these for pytest as dummy data
# def JSON_to_pandas():
#     dummy = {
#         'ID': [11, 12, 13, 14, 15, 16, 17, 18],
#         'Latitude': [-32.040650, -32.010274, -32.090316, -32.000879, -31.900399, -31.899364, -20, -10],
#         'Longitude': [115.905166, 115.886444, 115.870573, 115.920247, 115.799830, 115.801288, 20, 10]
#     }

#     dummy2 = {
#         'ID': [16, 12, 13, 14, 15, 11, 17, 18],
#         'Latitude': [-31.899364, -32.010274, -32.090316, -32.000879, -31.900399, -32.040650, -20, -10],
#         'Longitude': [115.801288, 115.886444, 115.870573, 115.920247, 115.799830, 115.905166, 20, 10]
#     }

#     df = pd.DataFrame(dummy)
#     return df

# def JSON_to_pandas2():
#     dummy2 = {
#         'ID': [16, 12, 13, 14, 15, 11, 17, 18],
#         'Latitude': [-31.899364, -32.010274, -32.090316, -32.000879, -31.900399, -32.040650, -20, -10],
#         'Longitude': [115.801288, 115.886444, 115.870573, 115.920247, 115.799830, 115.905166, 20, 10]
#     }

#     df = pd.DataFrame(dummy2)
#     return df

# k = 3
# split_threshold = 2
# dm = DistanceMatrixContext(SpatialMatrix())
# rs = RouteSolverContext(BruteForceSolver())
# optimise_route(None, k, split_threshold, dm, rs)
