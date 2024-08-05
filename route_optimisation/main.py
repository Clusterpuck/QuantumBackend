from customer_allocation.customer_allocation import get_customer_allocation, create_dictionary
from route_partitioning import *
from geographic_processing import geographic_array
from distance_matrix.distance_matrix_context import DistanceMatrixContext
import pandas as pd

from distance_matrix.spatial_matrix import SpatialMatrix
from route_solver.route_solver_context import RouteSolverContext
from route_solver.brute_force_solver import BruteForceSolver


dataset = {
  'ID': [11, 12, 13, 14, 15, 16],
  'Customer': ["Woolworths Riverton", "Coles Karawara", "Spud Shed Jandakot", "Spud Shed Bentley", "Woolworths Innaloo", "Spud Shed Innaloo"]
}

runsheet = pd.DataFrame(dataset)

k = 3
#TODO: Make this a function
if __name__=="__main__": 
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