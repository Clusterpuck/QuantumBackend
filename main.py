from customer_allocation import get_customer_allocation, create_dictionary
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

testdataset = {
  'ID': [11, 12, 16, 14, 15, 13]
}

id_list = [11, 12, 16, 14, 15, 13]
df = pd.DataFrame({'ID': id_list})


testdataset = pd.DataFrame(testdataset)
runsheet = pd.DataFrame(dataset)


k = 3

if __name__=="__main__": 
    allocation_array = get_customer_allocation(runsheet, k)
    runsheet_dictionary = create_dictionary(runsheet)
    print(allocation_array)
    tree = None
    tree = partition_routes2(allocation_array, 2, runsheet_dictionary)
    print(allocation_array)
    print(tree)
    connection_string = os.getenv('QuantumTestString')
    print(geographic_array(df,connection_string))

    #WORKS
    #context = DistanceMatrixContext(SpatialMatrix())
    #context.build_leaf_matrix(tree)
    #context.build_parent_matrix(tree)

    dm = DistanceMatrixContext(SpatialMatrix())
    rs = RouteSolverContext(BruteForceSolver())
    tree.post_order_dfs2(dm, rs)


    #tree.solve_tree()
    #print(runsheet_dictionary)

    
    # Check route for splitting                                                 (ROUTE PARTITIONING)
    # IF statement to check each individual route for additional splitting      (ROUTE PARTITIONING)
    
    # V2
    # (Create Distance Matrix)
    # Small enough route?                                                       (ROUTE SOLVER)
    #   Get our distance matrix, Comes from MapBox                              
    #   Send to QTSP
    #   Decipher answer
    # Large Route?                                                              (ROUTE SOLVER)
    #   Make a param for number of clusters, work out better method later, Dist.
    #   Solve the smaller clusters, Quantum
    #   Link starts and ends of each cluster
    #   Solve the bigger TSP, Quantum
    #   Decipher and combine large routes
    # Package them in some way for webapp to read?

    # From a tree perspective
    # Post Order Setup
      # If leaf, make distance matrix
    
    # Post Order Solution
      # If leaf, 
        # Make Matrix
        # Solve Matrix
      # If Parent,
        # Make Matrix
          # Grab start and end of each child
        # Solve Matrix