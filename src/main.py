import math
import numpy as np
from sklearn.cluster import KMeans

from pydantic_models import CartesianOrder, Order
# from distance_matrix.distance_matrix_context import DistanceMatrixContext
from route_optimisation.route_solver.brute_force_solver import BruteForceSolver
from route_optimisation.route_solver.route_solver_context import RouteSolverContext


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


def cartesian_cluster(orders: list[CartesianOrder], k: int):
    points = [[o.x, o.y, o.z] for o in orders]
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(points)
    return kmeans.labels_


def solve_cluster(
    orders: list[CartesianOrder],
    distance_matrix,#: DistanceMatrixContext,
    route_solver: RouteSolverContext,
) -> list[CartesianOrder]:
    """
    Find the optimal route and cost for a node.

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

    # Solve and use to rearrange input orders
    optimal_indices, _ =  route_solver.solve(matrix)  # TODO: Drop cost for now, will use later
    optimal_route = [orders[i] for i in optimal_indices]
    return optimal_route


def solve_asym_tsp(  # TODO: Need to merge this with the other solver
    nodes: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    distance_matrix,#: DistanceMatrixContext,
    route_solver: RouteSolverContext,
) -> list[int]:
    """
    Find the optimal route and cost for a node. Needs to handle asym, so needs start and end rather than cartesian form
    converts pairs of start/end to distance matrix, and returns relative indices of new order?

    Nodes can be entire subroutes, so asymetric distances are calculated as a node's end coord to other nodes' start coords.
    To represent single point nodes (symmetric TSP), set start coords == end coords.

    Parameters
    ----------
    nodes: list of nodes
        Unordered list of "TSP cities", each with start and end coordinates.
    distance_matrix: DistanceMatrix
        Strategy to create the distance matrix
    route_solver: RouteSolver
        Strategy to finding the optimal route

    Returns
    -------
    list of int
        Optimal visit order, labelled by relative indices
    """
    # NOTE: The difference between cartesian solve and traffic is that one can use the precomputed xyz, while the other much fetch via api
    # TODO: Will move new implementation into strategy once approved

    n = len(nodes)
    matrix = np.zeros([n] * 2)

    # Populate asymmetric distance matrix
    for i in range(n):
        for j in range(n):
            if i != j:
                node_a = nodes[i][1]  # End of route 1
                node_b = nodes[j][0]  # Start of route 2
                matrix[i, j] = math.dist(node_a, node_b)

    # Solve and use to rearrange input orders
    optimal_indices, _ =  route_solver.solve(matrix)  # TODO: Drop cost for now, will use later
    return optimal_indices


def flatten_list(deep_list):
    # Arbitrary list flattening generator
    for i in deep_list:
        if isinstance(i, list):
            yield from flatten_list(i)
        else:
            yield i


def solve_routes(
    orders: list[CartesianOrder],
    split_threshold: int,
    dm,#: DistanceMatrixContext,
    rs: RouteSolverContext,
) -> tuple[list[CartesianOrder | list], tuple[int, int, int], tuple[int, int, int]]:
    # Base case: Safe size to solve
    if len(orders) <= split_threshold:
        # Solve symmetric TSP (single point nodes)
        nodes = [((o.x, o.y, o.z), (o.x, o.y, o.z)) for o in orders]
        solved_labels = solve_asym_tsp(nodes, dm, rs)

        # Reorder order objects by solution
        base_solution = [orders[new_i] for new_i in solved_labels]
        print(f"solved: {[o.order_id for o in base_solution]}")

        # Extract start/end coords for the parent's convenience
        start = base_solution[0]
        end = base_solution[-1]
        start_coords = (start.x, start.y, start.z)
        end_coords = (end.x, end.y, end.z)

        return base_solution, start_coords, end_coords

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
        subtree, start_coords, end_coords = solve_routes(subset_orders, split_threshold, dm, rs)
        unsolved_branches.append(subtree)
        spoofed_nodes.append((start_coords, end_coords))
    
    # Solve current cluster's TSP using the asymmetric node data...
    solved_labels = solve_asym_tsp(spoofed_nodes, dm, rs)

    # ...And encode this into the solution tree via branch ordering
    solution_tree = [unsolved_branches[new_i] for new_i in solved_labels]

    print(f"stitch order: {solved_labels}, progress so far: {[o.order_id for o in flatten_list(solution_tree)]}")

    # Finally, spoof next TSP node by attaching route start/end data
    # Aka start coords of the spoofed route's start, and opposite case for end
    start_coords = spoofed_nodes[solved_labels.index(0)][0]
    end_coords = spoofed_nodes[solved_labels.index(len(solved_labels) - 1)][1]

    return solution_tree, start_coords, end_coords


def partition_vehicles(
    orders: list[Order],
    vehicles: int,
    split_threshold: int,
    dm,#: DistanceMatrixContext,
    rs: RouteSolverContext,
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
    for cluster in np.unique(cluster_labels):
        order_indices = np.where(cluster_labels == cluster)[0]
        subset_orders = [orders[i] for i in order_indices]

        # Start up recursion for each vehicle
        route_tree, _, _ = solve_routes(subset_orders, split_threshold, dm, rs)

        # If subdivided from recursion, flatten into final routes
        vehicle_routes.append([o for o in flatten_list(route_tree)])

    # But will also return raws for display
    return vehicle_routes, route_tree


# def subdivide_routes(
#     orders: list[CartesianOrder],
#     split_threshold: int,
#     dm,#: DistanceMatrixContext,
#     rs: RouteSolverContext,
# ) -> tuple[list[CartesianOrder], list]:
#     # Split oversized clusters further
#     labels = cartesian_cluster(orders, split_threshold)
#     # Aligned cluster labels

#     print(f"sub cluster: {[o.order_id for o in orders]}")
#     print(f"labels: {labels}")

#     # Try solving each cluster
#     optimal_route = []
#     tree_reconstruction = []
#     for cluster in np.unique(labels):
#         order_indices = np.where(labels == cluster)[0]
#         subset_orders = [orders[i] for i in order_indices]
#         print(f"considering: {[o.order_id for o in subset_orders]}")
#         if len(subset_orders) > split_threshold:
#             # Too big, subdivide further
#             print("subdivide")
#             subset_orders = [orders[i] for i in order_indices]
#             subdivide_routes(subset_orders, split_threshold, dm, rs)
#         else:
#             # Safe size to solve
#             print("solve")

#             # Extract start/ends to spoof TSP nodes
#             nodes = [for o in orders]
#             solved_child = solve_asym_tsp(orders, dm, rs)
#             [orders]  # Match indices back to orders
#             optimal_route.extend(solved_child)

#             # Since it's useful for debugging, build the tree in reverse
#             tree_reconstruction.append(orders)

#     return optimal_route, tree_reconstruction


# def partition_vehicles(
#     orders: list[Order],
#     vehicles: int,
#     split_threshold: int,
#     dm,#: DistanceMatrixContext,
#     rs: RouteSolverContext,
# ) -> tuple[list[list[CartesianOrder]], list]:
#     # The main entry wrapper

#     # Validate params
#     # The presence of pydantic means only range checking is required
#     seen_id = set()
#     for order in orders:
#         if order.order_id in seen_id:
#             raise ValueError("order_id must be unique")
#         elif not (-90 <= order.lat <= 90):
#             raise ValueError("lat must be between -90 and 90")
#         elif not (-180 <= order.long <= 180):
#             raise ValueError("long must be between -180 and 180")
        
#         seen_id.add(order.order_id)

#     # Pre-compute Cartesians
#     orders = orders_to_cartesian(orders)

#     # Preliminary cluster into distinct vehicle routes
#     labels = cartesian_cluster(orders, vehicles)

#     print(f"vehicle cluster: {[o.order_id for o in orders]}")
#     print(f"labels: {labels}")

#     # TODO: This duplication can be compressed further
#     # Try solving each cluster
#     optimal_route = []
#     tree_reconstruction = []
#     for cluster in np.unique(labels):
#         order_indices = np.where(labels == cluster)[0]
#         subset_orders = [orders[i] for i in order_indices]
#         print(f"considering: {[o.order_id for o in subset_orders]}")
#         if len(subset_orders) > split_threshold:
#             # Too big, subdivide further
#             print("subdivide")
#             subset_orders = [orders[i] for i in order_indices]
#             subdivide_routes(subset_orders, split_threshold, dm, rs)
#         else:
#             # Safe size to solve
#             print("solve")
#             solved_child = solve_cluster(orders, dm, rs)
#             optimal_route.extend(solved_child)

#             # Since it's useful for debugging, build the tree in reverse
#             tree_reconstruction.append(orders)

#     return optimal_route, tree_reconstruction


if __name__ == "__main__":
    dummy = {
        'order_id': [11, 12, 13, 14, 15, 16, 17, 18],
        'lat': [-32.040650, -32.010274, -32.090316, -32.000879, -31.900399, -31.899364, -20, -10],
        'long': [115.905166, 115.886444, 115.870573, 115.920247, 115.799830, 115.801288, 20, 10]
    }

    # dummy2 = {
    #     'ID': [16, 12, 13, 14, 15, 11, 17, 18],
    #     'Latitude': [-31.899364, -32.010274, -32.090316, -32.000879, -31.900399, -32.040650, -20, -10],
    #     'Longitude': [115.801288, 115.886444, 115.870573, 115.920247, 115.799830, 115.905166, 20, 10]
    # }

    #  flip dict[list] to list[dict], then convert to list[Order]
    new_orders = [Order(**dict(zip(dummy,t))) for t in zip(*dummy.values())]
    print(new_orders)

    # test code
    rs = RouteSolverContext(BruteForceSolver())
    optimal_route_per_vehicle, raw_solution_tree = partition_vehicles(
        new_orders, 2, 3, None, rs
    )

    print()
    print(f"solution: {[o.order_id for v in optimal_route_per_vehicle for o in v]}")
    print(raw_solution_tree)


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
