"""Abstract class for route solver strategies"""

from abc import ABC, abstractmethod

import numpy as np


class RouteSolver(ABC):
    """Abstract class for route solver strategies"""

    @abstractmethod
    def solve(self, distance_matrix: np.ndarray) -> tuple[list[int], int]:
        """
        Solve a distance matrix

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
        pass
