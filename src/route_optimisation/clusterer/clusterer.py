"""Abstract class for clustering strategies"""

from abc import ABC, abstractmethod

import numpy as np

from pydantic_models import Order


class Clusterer(ABC):
    """Abstract class for clustering strategies"""

    # Look, Google's API thinks it's a valid word, so it's fine
    @abstractmethod
    def cluster(self, orders: list[Order]) -> np.ndarray:
        """
        Do clustering on a list of orders

        Parameters
        ----------
        orders : list of Order
            List of orders

        Returns
        -------
        ndarray
            1D, input-aligned array labelled by their cluster
            (eg. [0, 0, 1, 2, 0]).
        """
        pass
