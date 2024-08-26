"""Factory for distance metric"""

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
    """
    Factory class for creating a particular DistanceFinder instance
    """

    def create(self, distance_type: str) -> DistanceFinder:
        """
        Validate and decide what distance metric the solver will use for the API

        Parameters
        ----------
        distance_type : str
            distace metric to be used

        Returns
        -------
        DistanceFinder
            DistanceFinder strategy depending on distance_type

        Raises
        ------
        ValueError
            If provided distance_type doesn't exist.
        """
        if distance_type == "cartesian":
            return CartesianDistanceFinder()
        elif distance_type == "mapbox":
            return MapboxDistanceFinder(
                requester=MapboxRequester(token=os.environ["MAPBOX_TOKEN"])
            )
        else:
            raise ValueError("Unsupported distance type.")
