"""Find the optimal route via brute forcing checking every valid possibility"""

from itertools import permutations
import numpy as np
from route_optimisation.route_solver.route_solver import RouteSolver

class BruteForceSolver(RouteSolver):
    """
    Generates every permutation of routes and finds the optimal route

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
    def solve(self, distance_matrix: np.ndarray) -> tuple[list[int], int]:
        n = len(distance_matrix)
        index = list(range(n))
        all_routes = permutations(index)

        lowest_cost = float('inf')
        best_route = []
        for route in all_routes:
            current_cost = self.__get_route_cost(route, distance_matrix)
            if current_cost < lowest_cost:
                lowest_cost = current_cost
                best_route = route
        return list(best_route), current_cost

    def __get_route_cost(self, route, distance_matrix):
        cost = 0
        # Range from 0 to size of route (no need to add cost of last node)
        order = range(len(route) - 1)

        for i in order:
            # Add cost of current node and path to the next
            cost += distance_matrix[route[i]][route[i + 1]] 
        return cost
