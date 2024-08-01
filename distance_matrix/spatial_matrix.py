from distance_matrix.distance_matrix import DistanceMatrix
from cluster_hierarchy_tree import TreeNode
import numpy as np

class SpatialMatrix(DistanceMatrix):

    def build_parent_matrix(self, node: TreeNode):
        # Implement matrix creation
        print("Building SpatialMatrix for parent")
        # query for lat, long for start and end. store as [x_start, x_end, y_start, y_end, ...]? or Seperate? works either way
            # Geographic processing, geographic array
            # geographic_to_cartesian
        # loop over start coords
            # loop over end coords
                # Get cartesian distance (A + A_end -> B_start)
                # If i==j, set to 0
        # return matrix


    def build_leaf_matrix(self, node):
        # Implement matrix creation
        print("Building SpatialMatrix for leaf")
        customers = node.get_customers()
        n = len(customers)
        np.zeros((n, n), dtype=float) # Create n x n zero-filled array
        # Query for lat, long for each point
            # Geographic processing, geographic array
            # geographic_to_cartesian
        # Double for loop over coords
            # Find cartesian distance (A + A_end -> B_start)
            # If i==j, set to 0
        # return matrix