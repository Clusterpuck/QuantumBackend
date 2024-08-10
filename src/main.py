import math
import numpy as np
from sklearn.cluster import KMeans

from pydantic_models import CartesianOrder, Order

from route_optimisation.distance_matrix.distance_finder import DistanceFinder
from route_optimisation.distance_matrix.cartesian_distance_finder import CartesianDistanceFinder
from route_optimisation.route_solver.brute_force_solver import BruteForceSolver
from route_optimisation.route_solver.route_solver import RouteSolver


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
                order_id=order.order_id,
                lat=order.lat,
                long=order.long,
                x=r * np.cos(r_lat) * np.cos(r_lon),
                y=r * np.cos(r_lat) * np.sin(r_lon),
                z=r * np.sin(r_lat),
            )
        )
    return cartesian_orders


# NOTE: Might be good to dedicate a strategy pattern to this in the future
def cartesian_cluster(orders: list[CartesianOrder], k: int) -> np.ndarray:
    points = [[o.x, o.y, o.z] for o in orders]
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(points)
    return kmeans.labels_


def flatten_list(deep_list):
    # Arbitrary list flattening generator
    for i in deep_list:
        if isinstance(i, list):
            yield from flatten_list(i)
        else:
            yield i


# NOTE: Could be moved into a RouteSolver strategy in the future, primed with internal solvers
# Something like RecursiveSolver(SpatialMatrixConverter(), BruteForceSolver()))
# However, it is currently returning raw_tree, which is useful
# Also would need to update other solvers with basic validation


def solve_routes(
    orders: list[CartesianOrder],
    split_threshold: int,
    dm: DistanceFinder,
    rs: RouteSolver,
) -> tuple[list[CartesianOrder | list], CartesianOrder, CartesianOrder]:
    # Base case: Safe size to solve
    if len(orders) <= split_threshold:
        # Solve symmetric TSP (single point nodes)
        nodes = [(o, o) for o in orders]
        distance_matrix = dm.build_matrix(nodes)
        solved_labels, _ = rs.solve(distance_matrix)
        # TODO: Drop cost for now, might use later if strategised

        # Reorder order objects by solution
        base_solution: list[CartesianOrder] = [orders[new_i] for new_i in solved_labels]
        print(f"solved: {[o.order_id for o in base_solution]}")

        # Extract start/end orders for the parent's TSP convenience
        route_start = base_solution[0]
        route_end = base_solution[-1]

        return base_solution, route_start, route_end

    # Recusion case: Split oversized cluster, recurse, solve cluster TSP
    print(f"sub cluster: {[o.order_id for o in orders]}")
    spoofed_nodes = []
    unsolved_branches = []

    # Cluster and wait for subcluster TSP solutions
    cluster_labels = cartesian_cluster(orders, split_threshold)
    print(f"sub cluster labels: {cluster_labels}")
    for cluster in np.unique(cluster_labels):
        order_indices = np.where(cluster_labels == cluster)[0]
        subset_orders = [orders[i] for i in order_indices]

        # Recursely gather solution subtrees and next TSP data
        subtree, subroute_start, subroute_end = solve_routes(
            subset_orders, split_threshold, dm, rs
        )
        unsolved_branches.append(subtree)
        spoofed_nodes.append((subroute_start, subroute_end))

    # Solve current cluster's TSP using the asymmetric node data...
    distance_matrix = dm.build_matrix(spoofed_nodes)
    solved_labels, _ = rs.solve(distance_matrix)
    # TODO: Drop cost for now, might use later if strategised

    # ...And encode this into the solution tree via branch ordering
    solution_tree = [unsolved_branches[new_i] for new_i in solved_labels]

    print(
        f"stitch order: {solved_labels}, progress so far: {[o.order_id for o in flatten_list(solution_tree)]}"
    )

    # Finally, spoof next TSP node by attaching route start/end data
    # Aka start coords of the spoofed route's start, and opposite case for end
    route_start = spoofed_nodes[solved_labels.index(0)][0]
    route_end = spoofed_nodes[solved_labels.index(len(solved_labels) - 1)][1]

    return solution_tree, route_start, route_end


def partition_vehicles(
    orders: list[Order],
    vehicles: int,
    split_threshold: int,
    dm,  #: DistanceMatrixMaker,
    rs: RouteSolver,
) -> tuple[list[list[CartesianOrder]], list[list[CartesianOrder | list]]]:
    # The main entry wrapper

    # Validate params
    # The presence of pydantic means only range checking is required
    seen_id = set()
    for order in orders:
        if order.order_id in seen_id:
            raise ValueError("order_id must be unique")
        elif not (-90 <= order.lat <= 90):
            raise ValueError("lat must be between -90 and 90")
        elif not (-180 <= order.long <= 180):
            raise ValueError("long must be between -180 and 180")

        seen_id.add(order.order_id)

    # Pre-compute Cartesians
    orders = orders_to_cartesian(orders)

    # Cluster into distinct vehicle routes
    vehicle_routes = []
    cluster_labels = cartesian_cluster(orders, vehicles)

    print(f"vehicle cluster: {[o.order_id for o in orders]}")
    print(f"labels: {cluster_labels}")

    # Gather each solution subtrees and TSP data (recurse as much as needed)
    # Start up recursion for each vehicle
    vehicle_routes = []
    raw_tree = []
    for cluster in np.unique(cluster_labels):
        order_indices = np.where(cluster_labels == cluster)[0]
        subset_orders = [orders[i] for i in order_indices]

        # Start up recursion for each vehicle
        route_tree, _, _ = solve_routes(subset_orders, split_threshold, dm, rs)
        raw_tree.append(route_tree)

        # If subdivided from recursion, flatten into final routes
        vehicle_routes.append([o for o in flatten_list(route_tree)])

    # But will also return raws for display
    return vehicle_routes, raw_tree


if __name__ == "__main__":
    # dummy = {
    #     "order_id": [11, 12, 13, 14, 15, 16, 17, 18],
    #     "lat": [
    #         -32.040650,
    #         -32.010274,
    #         -32.090316,
    #         -32.000879,
    #         -31.900399,
    #         -31.899364,
    #         -20,
    #         -10,
    #     ],
    #     "long": [
    #         115.905166,
    #         115.886444,
    #         115.870573,
    #         115.920247,
    #         115.799830,
    #         115.801288,
    #         20,
    #         10,
    #     ],
    # }

    dummy = {
        "order_id": [16, 12, 13, 14, 15, 11, 17, 18],
        "lat": [
            -31.899364,
            -32.010274,
            -32.090316,
            -32.000879,
            -31.900399,
            -32.040650,
            -20,
            -10,
        ],
        "long": [
            115.801288,
            115.886444,
            115.870573,
            115.920247,
            115.799830,
            115.905166,
            20,
            10,
        ],
    }

    #  Flip dict[list] to list[dict], then convert to list[Order]
    orders = [Order(**dict(zip(dummy, t))) for t in zip(*dummy.values())]
    print(orders)

    # Define strategies
    dm = CartesianDistanceFinder()
    rs = BruteForceSolver()

    optimal_route_per_vehicle, raw_solution_tree = partition_vehicles(
        orders, 2, 3, dm, rs
    )

    print()
    print(f"solution: {[o.order_id for v in optimal_route_per_vehicle for o in v]}")


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
