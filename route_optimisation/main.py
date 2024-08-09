import pandas as pd

from customer_allocation.customer_assignment import get_customer_allocation, create_dictionary
from route_partitioning import partition_routes
from distance_matrix.distance_matrix_context import DistanceMatrixContext

from distance_matrix.spatial_matrix import SpatialMatrix
from route_solver.route_solver_context import RouteSolverContext
from route_solver.brute_force_solver import BruteForceSolver

def optimise_route(data, k, split_threshold, dm, rs):
    delivery_list = JSON_to_pandas(data)
    delivery_list2 = JSON_to_pandas2()
    allocation_array = get_customer_allocation(delivery_list, k, split_threshold)
    if(allocation_array is not None):
        delivery_dictionary = create_dictionary(delivery_list)
        tree = partition_routes(allocation_array, split_threshold, delivery_dictionary, dm, rs)

        print(tree)
        #print(allocation_array)
    print("=====================================")
    allocation_array = get_customer_allocation(delivery_list2, k, split_threshold)
    if(allocation_array is not None):
        delivery_dictionary = create_dictionary(delivery_list2)
        tree = partition_routes(allocation_array, split_threshold, delivery_dictionary, dm, rs)

        print(tree)
        #print(allocation_array)


def JSON_to_pandas(data):
    dummy = {
        'ID': [11, 12, 13, 14, 15, 16, 17, 18],
        'Latitude': [-32.040650, -32.010274, -32.090316, -32.000879, -31.900399, -31.899364, -20, -10],
        'Longitude': [115.905166, 115.886444, 115.870573, 115.920247, 115.799830, 115.801288, 20, 10]
    }

    dummy2 = {
        'ID': [16, 12, 13, 14, 15, 11, 17, 18],
        'Latitude': [-31.899364, -32.010274, -32.090316, -32.000879, -31.900399, -32.040650, -20, -10],
        'Longitude': [115.801288, 115.886444, 115.870573, 115.920247, 115.799830, 115.905166, 20, 10]
    }

    df = pd.DataFrame(dummy)
    return df

def JSON_to_pandas2():
    dummy2 = {
        'ID': [16, 12, 13, 14, 15, 11, 17, 18],
        'Latitude': [-31.899364, -32.010274, -32.090316, -32.000879, -31.900399, -32.040650, -20, -10],
        'Longitude': [115.801288, 115.886444, 115.870573, 115.920247, 115.799830, 115.905166, 20, 10]
    }

    df = pd.DataFrame(dummy2)
    return df

k = 3
split_threshold = 2
dm = DistanceMatrixContext(SpatialMatrix())
rs = RouteSolverContext(BruteForceSolver())
optimise_route(None, k, split_threshold, dm, rs)