from itertools import permutations
from route_optimisation.route_solver.route_solver import RouteSolver
import numpy as np

class BruteForceSolver(RouteSolver):

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
        order = range(len(route) - 1) # Range from 0 to size of route (no need to add cost of last node)

        for i in order:
            cost += distance_matrix[route[i]][route[i + 1]] # Add cost of current node and path to the next
        return cost

