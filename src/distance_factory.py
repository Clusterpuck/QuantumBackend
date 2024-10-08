import os

from route_optimisation.distance_matrix.cartesian_distance_finder import (
    CartesianDistanceFinder,
)
from route_optimisation.distance_matrix.mapbox_distance_finder import (
    MapboxDistanceFinder,
    MapboxRequester,
)
from route_optimisation.distance_matrix.distance_finder import DistanceFinder


class DistanceFactory:
    # Validate and decide what distance metric the solver will use for the API
    def create(self, distance_type: str) -> DistanceFinder:
        if distance_type == "cartesian":
            return CartesianDistanceFinder()
        elif distance_type == "mapbox":
            return MapboxDistanceFinder(
                requester=MapboxRequester(token=os.environ["MAPBOX_TOKEN"])
            )
        else:
            raise ValueError("Unsupported distance type.")
