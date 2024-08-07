import pandas as pd

from customer_allocation.customer_assignment import get_customer_allocation, create_dictionary
from route_partitioning import partition_routesV3
from distance_matrix.distance_matrix_context import DistanceMatrixContext

from distance_matrix.spatial_matrix import SpatialMatrix
from route_solver.route_solver_context import RouteSolverContext
from route_solver.brute_force_solver import BruteForceSolver

def new_main():
    delivery_list = JSON_to_pandas()
    k = 3
    dm = DistanceMatrixContext(SpatialMatrix())
    rs = RouteSolverContext(BruteForceSolver())

    allocation_array = get_customer_allocation(delivery_list, k)
    if(allocation_array is None):
        print("Error")
    else:
        runsheet_dictionary = create_dictionary(delivery_list)
        tree = partition_routesV3(allocation_array, 1, runsheet_dictionary, dm, rs)

        print(tree)
        print(allocation_array)


def JSON_to_pandas():
    dummy = {
        'ID': [11, 12, 13, 14, 15, 16],
        'Latitude': [-32.040650, -32.010274, -32.090316, -32.000879, -31.900399, -31.899364],
        'Longitude': [115.905166, 115.886444, 115.870573, 115.920247, 115.799830, 115.801288]
    }

    df = pd.DataFrame(dummy)
    return df
    # check later https://saturncloud.io/blog/how-to-convert-nested-json-to-pandas-dataframe-with-specific-format/

#TODO: Make this a function
#TODO: Correct terminology. A runsheet exists for each truck
if __name__=="__main__": 
    new_main()