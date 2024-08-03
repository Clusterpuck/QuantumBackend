from abc import ABC, abstractmethod

class DistanceMatrix(ABC):
    @abstractmethod
    def build_parent_matrix(self, node):
        pass
    @abstractmethod
    def build_leaf_matrix(self, node):
        pass

    