from typing import Literal
import numpy as np
from urllib3.exceptions import MaxRetryError
from requests.adapters import HTTPAdapter, Retry
import requests

from route_optimisation.distance_matrix.distance_finder import DistanceFinder
from pydantic_models import Order

# Assumed driving and always stops at curb (aka correct side of road)
ENDPOINT = "https://api.mapbox.com/directions-matrix/v1/mapbox/driving/"
APPROACH = "curb"


class MapboxDistanceFinder(DistanceFinder):
    def __init__(self, token: str, max_retries: int = 5):
        """
        Inits an asymmetric distance matrix maker that uses a Mapbox's Matrix
        API. For this project's scope, an semi-arbitrary cap of 12 is used.

        "Distances" use drive duration in seconds, which can be asymmetric.
        Note this is not live traffic. Failure to find a path (unroutable
        locations) will raise an exception.

        Parameters
        ----------
        token : str
            A Mapbox public access token (i.e. "pk.[long-token]").
        max_retries : int, default=5
            Max retries before giving up. 5 retries should be enough to fully
            avoid rate limiting, but takes a literal good minute.
        """
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self.__token = token
        self.__max_retries = max_retries

    def __construct_request(self, nodes: list[tuple[Order, Order]]) -> str:
        # Just do it the lazy way for now, limiting to 12x12 even on symmetric
        locations = [f"{node[0].lat},{node[0].lon}"]
        for node in nodes:
            locations.append(f"{node[1].lat},{node[1].lon}")
        locations_query = ";".join(locations)
        approach_query = ";".join([APPROACH] * len(nodes))

        # Select first/second halves as source/dest sets
        sources_query = ";".join([str(x) for x in range(len(nodes))])
        dests_query = ";".join([str(x) for x in range(len(nodes), len(nodes) * 2)])

        return f"{ENDPOINT}{locations_query}?approaches={approach_query}&sources={sources_query}&destinations={dests_query}&access_token={self.__token}"

    def build_matrix(self, nodes: list[tuple[Order, Order]]) -> np.ndarray:
        """
        Converts TSP nodes into a distance matrix. To create an asymmetric
        matrix, set the start and end coords per "node" to differ.

        Mapbox has a cap of 25 lat/lon inputs. For simplicity, start and ends
        will be capped to 12 nodes each.

        Parameters
        ----------
        nodes: list of (start_order, end_order)
            Unordered list of "TSP cities", each with start and end coordinates as orders.

        Returns
        -------
        ndarray
            Asymmetric distance matrix

        Raises
        ------
        ValueError
            Node count is not between 1-12 inclusive.
        MaxRetryError
            Mapbox cannot find a valid solution within the max retries.
        RuntimeError
            Node set contains unroutable points.
        """
        if not (0 < len(nodes) <= 12):
            raise ValueError("Node count must be between 1 and 12.")

        if (
            len(nodes) == 1
            and nodes[0][0].lat == nodes[0][1].lat
            and nodes[0][0].lon == nodes[0][1].lon
        ):
            # Trivial symmetric case, always 2D containing a single 0
            return np.array([1] * 2)
        else:
            # Set up exp backoff (starting at 2s) to resist rate limiting
            # Limit seems to be 60s, returning code 429
            with requests.Session() as session:
                retry_adapter = HTTPAdapter(
                    max_retries=Retry(
                        total=self.__max_retries,
                        backoff_factor=2,
                        backoff_max=60,
                        status_forcelist=[429],
                    )
                )
                session.mount("https://", retry_adapter)
                req = session.get(self.__construct_request(nodes))
            # TODO: Investigate if there is a retry-after header later on
            # We will let any errors after this point bubble up for now...
            # TODO: Wrap a proper error type around value/runtime/this and return to client

            matrix_data = req.json()["durations"]
            distance_matrix = np.array(matrix_data, dtype="float64")

            # Also check that it is fully routable
            if np.isnan(distance_matrix).any():
                raise RuntimeError("Nodes contain unroutable points using Mapbox.")

            return distance_matrix
