"""Abstract class for distance finder strategies"""

from abc import ABC, abstractmethod

import numpy as np

from pydantic_models import Order


class DistanceFinder(ABC):
    """Abstract class for distance finder strategies"""

    @abstractmethod
    def build_matrix(self, nodes: list[tuple[Order, Order]]) -> np.ndarray:
        """
        Create a distance matrix

        Parameters
        ----------
        nodes: list of (start_order, end_order)
            Unordered list of "TSP cities", each with start and end coordinates as orders.

        Returns
        -------
        ndarray
            Asymmetric distance matrix
        """
        pass
