import numpy as np
import math

from route_optimisation.distance_matrix.distance_finder import DistanceFinder
from pydantic_models import Order


class CartesianDistanceFinder(DistanceFinder):
    def build_matrix(self, nodes: list[tuple[Order, Order]]) -> np.ndarray:
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

        if not isinstance(nodes, list):
            raise TypeError("The 'nodes' parameter should be of type List[Tuple[Order, Order]]")
        
        if not all(isinstance(item, tuple) and len(item) == 2 for item in nodes):
            raise TypeError("Each item in 'nodes' should be a tuple of two Orders")
        
        if not all(isinstance(item[0], Order) and isinstance(item[1], Order) for item in nodes):
            raise TypeError("Each element of the tuple in 'nodes' should be of type Order")

        if not nodes:
            raise ValueError("The 'nodes' list cannot be empty")

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
