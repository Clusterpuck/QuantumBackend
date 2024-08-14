from abc import ABC, abstractmethod

import numpy as np

from pydantic_models import Order


class Clusterer(ABC):
    # Look, Google's API thinks it's a valid word, so it's fine
    @abstractmethod
    def cluster(self, orders: list[Order]) -> np.ndarray:
        pass
