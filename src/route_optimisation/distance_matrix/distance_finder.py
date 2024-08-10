from abc import ABC, abstractmethod

import numpy as np

from pydantic_models import Order


class DistanceFinder(ABC):
    @abstractmethod
    def build_matrix(self, nodes: list[tuple[Order, Order]]) -> np.ndarray:
        pass
