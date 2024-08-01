from distance_matrix.distance_matrix import DistanceMatrix
from cluster_hierarchy_tree import TreeNode
import numpy as np

class SpatialMatrix(DistanceMatrix):

    def build_parent_matrix(self, node: TreeNode):
        # Implement matrix creation
        print("Building SpatialMatrix for parent")

    def build_leaf_matrix(self, node):
        # Implement matrix creation
        print("Building SpatialMatrix for leaf")
        customers = node.get_customers()
        n = len(customers)
        np.zeros((n, n), dtype=float) # Create n x n array