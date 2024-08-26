"""Find the optimal route via brute forcing checking every valid possibility"""

from itertools import permutations
import numpy as np
from route_optimisation.route_solver.route_solver import RouteSolver

class BruteForceSolver(RouteSolver):
    """BruteForceSolver inheriting from RouteSolver"""
    def solve(self, distance_matrix: np.ndarray) -> tuple[list[int], int]:
        """
        Generate every permutation of routes and find the optimal route

        Parameters
        ----------
        distance_matrix: ndarray
            2D, Asymmetric distance matrix

        Returns
        -------
        tuple[list[int], int]
            List[int] represents the list of the optimal route containing order_ids
            int represents the total cost of the route
        """
        lowest_cost = float('inf')
        best_route = []

        # Iterate over every possible permutation in matrix
        for route in permutations(list(range(len(distance_matrix)))):
            current_cost = self.__get_route_cost(route, distance_matrix)
            if current_cost < lowest_cost:
                lowest_cost = current_cost
                best_route = route
        return list(best_route), lowest_cost

    def __get_route_cost(self, route, distance_matrix) -> float | int:
        """
        Find the cost of the route

        Parameters
        ----------
        route: tuple[int, ...]
            The route being measured
        distance_matrix: ndarray
            2D, Asymmetric distance matrix

        Returns
        -------
        cost: float | int
            Cost of the route
        """
        cost = 0
        # Getting edges between nodes to get cost
        for i in range(len(route) - 1):
            # Add cost of current node and path to the next
            cost += distance_matrix[route[i]][route[i + 1]]
        return cost
