from route_optimisation.route_solver.route_solver import RouteSolver

class RouteSolverContext:
    def __init__(self, strategy: RouteSolver):
        self._strategy = strategy

    def set_strategy(self, strategy: RouteSolver):
        self._strategy = strategy

    def solve(self, distance_matrix):
        return self._strategy.solve(distance_matrix)