import numpy as np
import math

from route_optimisation.distance_matrix.distance_finder import DistanceFinder
from pydantic_models import CartesianOrder


class CartesianDistanceFinder(DistanceFinder):
    def build_matrix(self, nodes: list[tuple[CartesianOrder, CartesianOrder]]) -> np.ndarray:
        """
        Converts TSP nodes into a distance matrix. To create an asymmetric
        matrix, set the start and end coords per "node" to differ.

        Parameters
        ----------
        nodes: list of (start_order, end_order)
            Unordered list of "TSP cities", each with start and end coordinates as orders.

        Returns
        -------
        ndarray
            Asymmetric distance matrix
        """
        n = len(nodes)
        distance_matrix = np.zeros([n] * 2)

        # Populate asymmetric distance matrix
        for i in range(n):
            for j in range(n):
                if i != j:
                    node_a = nodes[i][1]  # End of route 1
                    node_b = nodes[j][0]  # Start of route 2
                    distance_matrix[i, j] = math.dist(
                        (node_a.x, node_a.y, node_a.z), (node_b.x, node_b.y, node_b.z)
                    )
        return distance_matrix
