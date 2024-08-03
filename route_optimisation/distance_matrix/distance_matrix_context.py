from distance_matrix.distance_matrix import DistanceMatrix

class DistanceMatrixContext:
    def __init__(self, strategy: DistanceMatrix):
        self._strategy = strategy

    def set_strategy(self, strategy: DistanceMatrix):
        self._strategy = strategy

    def build_parent_matrix(self, node):
        return self._strategy.build_parent_matrix(node)
    
    def build_leaf_matrix(self, node):
        return self._strategy.build_leaf_matrix(node)