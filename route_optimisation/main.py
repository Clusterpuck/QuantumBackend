from customer_allocation.customer_allocation import get_customer_allocation, create_dictionary, get_customer_allocationV2, create_dictionaryV2
from route_partitioning import *
from geographic_processing import geographic_array
from distance_matrix.distance_matrix_context import DistanceMatrixContext
import pandas as pd

from distance_matrix.spatial_matrix import SpatialMatrix
from route_solver.route_solver_context import RouteSolverContext
from route_solver.brute_force_solver import BruteForceSolver

def new_main():
    print("new_main")
    runsheet = JSON_to_pandas()
    k = 3
    allocation_array = get_customer_allocationV2(runsheet, k) # Works
    runsheet_dictionary = create_dictionaryV2(runsheet) # Works
    print(runsheet_dictionary) # Works
    tree = None
    tree = partition_routesV2(allocation_array, 2, runsheet_dictionary) # Works

    dm = DistanceMatrixContext(SpatialMatrix())
    rs = RouteSolverContext(BruteForceSolver())
    tree.post_order_dfs2(dm, rs)
    print(tree)
    print(allocation_array)


def JSON_to_pandas():
    print("Grab JSON object and turn it into a dataframe")
    df = pd.DataFrame(newdata)
    df['ID'] = df['ID'].astype(int)
    return df
    # check later https://saturncloud.io/blog/how-to-convert-nested-json-to-pandas-dataframe-with-specific-format/

"""dataset = {
  'ID': [11, 12, 13, 14, 15, 16],
  'Customer': ["Woolworths Riverton", "Coles Karawara", "Spud Shed Jandakot", "Spud Shed Bentley", "Woolworths Innaloo", "Spud Shed Innaloo"]
}"""

newdata = {
    'ID': [11, 12, 13, 14, 15, 16],
    'Latitude': [-32.040650, -32.010274, -32.090316, -32.000879, -31.900399, -31.899364],
    'Longitude': [115.905166, 115.886444, 115.870573, 115.920247, 115.799830, 115.801288]
}

"""runsheet = pd.DataFrame(dataset)

k = 3
"""
#TODO: Make this a function
#TODO: Correct terminology. A runsheet exists for each truck
if __name__=="__main__": 
    """
    allocation_array = get_customer_allocation(runsheet, k)
    runsheet_dictionary = create_dictionary(runsheet)
    print(allocation_array)
    tree = None
    tree = partition_routes2(allocation_array, 1, runsheet_dictionary)
    print(allocation_array)
    print(tree)
    connection_string = os.getenv('QuantumTestString')

    dm = DistanceMatrixContext(SpatialMatrix())
    rs = RouteSolverContext(BruteForceSolver())
    tree.post_order_dfs2(dm, rs)
    print(tree)
    print(type(runsheet))
    print(type(runsheet_dictionary))
    """
    new_main()