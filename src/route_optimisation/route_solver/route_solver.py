from abc import ABC, abstractmethod

import numpy as np

class RouteSolver(ABC):
    @abstractmethod
    def solve(self, distance_matrix: np.ndarray) -> tuple[list[int], int]:
        pass