from abc import ABC, abstractmethod

class RouteSolver(ABC):
    @abstractmethod
    def solve(self, distance_matrix):
        pass