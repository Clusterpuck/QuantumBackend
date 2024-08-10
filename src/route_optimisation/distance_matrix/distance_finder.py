from abc import ABC, abstractmethod

import numpy as np

from pydantic_models import CartesianOrder


class DistanceFinder(ABC):
    @abstractmethod
    def build_matrix(self, nodes: list[tuple[CartesianOrder, CartesianOrder]]) -> np.ndarray:
        pass
