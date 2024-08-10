import numpy as np
from sklearn.cluster import KMeans

from pydantic_models import Order, OrderInput

from route_optimisation.distance_matrix.distance_finder import DistanceFinder
from route_optimisation.distance_matrix.cartesian_distance_finder import CartesianDistanceFinder
from route_optimisation.route_solver.brute_force_solver import BruteForceSolver
from route_optimisation.route_solver.route_solver import RouteSolver


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
        r_lat, r_lon = np.deg2rad(order.lat), np.deg2rad(order.long)
        cartesian_orders.append(
            Order(
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
def cartesian_cluster(orders: list[Order], k: int) -> np.ndarray:
    """
    Clusters orders by their Cartesian distances, via K-means++.

    Parameters
    ----------
    orders: list of Order
        Contains all x, y, z points

    Returns
    -------
    ndarray
        1D, input-aligned array labelled by their cluster (eg. [0, 0, 1, 2, 0])
    """
    points = [[o.x, o.y, o.z] for o in orders]
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0, n_init=10).fit(points)
    return kmeans.labels_


def flatten_list(deep_list):
    # Arbitrary list flattening generator
    for i in deep_list:
        if isinstance(i, list):
            yield from flatten_list(i)
        else:
            yield i


# NOTE: Could move into a RouteSolver strategy, primed with internal solvers
# Init with RecursiveSolver(CartesianDistanceFinder(), BruteForceSolver()))
# However, needs to change solver interface, extract precompute/validation, etc
# So keep as a meta-solver for now...


def solve_routes(
    orders: list[Order],
    split_threshold: int,
    dm: DistanceFinder,
    rs: RouteSolver,
) -> tuple[list[Order | list], Order, Order]:
    # Base case: Safe size to solve
    if len(orders) <= split_threshold:
        # Solve symmetric TSP (single point nodes)
        nodes = [(o, o) for o in orders]
        distance_matrix = dm.build_matrix(nodes)
        solved_labels, _ = rs.solve(distance_matrix)
        # TODO: Drop cost for now, might use later if strategised

        # Reorder order objects by solution
        base_solution: list[Order] = [orders[new_i] for new_i in solved_labels]
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

    # Finally, spoof next TSP node by attaching route start/end data
    # Aka start coords of the spoofed route's start, and opposite case for end
    route_start = spoofed_nodes[solved_labels.index(0)][0]
    route_end = spoofed_nodes[solved_labels.index(len(solved_labels) - 1)][1]

    return solution_tree, route_start, route_end


def partition_vehicles(
    orders: list[Order],
    vehicles: int,
    split_threshold: int,
    dm: DistanceFinder,
    rs: RouteSolver,
) -> tuple[list[list[Order]], list[list[Order | list]]]:
    """
    Partition route into sub routes so they can be solved.

    Parameters
    ----------
    orders : list of Orders
        List of point allocations
    split_threshold : int
        How many clusters should exist for route partitioning
    delivery_dictionary : pandas.DataFrame
        Dataframes containing Customer IDs, latitude and longitude
    distance_matrix : DistanceMatrixContext
        The distance matrix building method
    route_solver : RouteSolver
        The route solving method

    Returns
    -------
    vehicle_routes : list of routes
        Unordered vehicle routes, each of which are ordered Orders
    raw_tree : deep list of minimal routes
        The clustering tree of routes, in solved order (ignoring vehicles)
    """
    # The main entry wrapper

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
    orders = [OrderInput(**dict(zip(dummy, t))) for t in zip(*dummy.values())]
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
