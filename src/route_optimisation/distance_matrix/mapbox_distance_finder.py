import numpy as np
import requests
import time

from route_optimisation.distance_matrix.distance_finder import DistanceFinder
from pydantic_models import Order

# Assumed driving and always stops at curb (aka correct side of road)
ENDPOINT = "https://api.mapbox.com/directions-matrix/v1/mapbox/driving/"
APPROACH = "curb"


class MapboxRequester:
    def __init__(self, token: str):
        """
        Inits the raw request builder for MapboxDistanceFinder. This inlcudes
        building the URL, calling, and the rate limiter handling logic.

        Designed to be mockable to allow testing everything but the API call.

        Parameters
        ----------
        token : str
            A Mapbox public access token (which starts with "pk").
        """
        self.__token = token

    def __construct_url(self, nodes: list[tuple[Order, Order]]) -> str:
        # Matrix API has a 25 input limit, inclusive of sources and dests
        # Though hugely suboptimal, just cap to 12 source-dest pairs
        locations = [f"{node[0].lon},{node[0].lat}"]  # Wants long first...
        for node in nodes:
            locations.append(f"{node[1].lon},{node[1].lat}")
        locations_query = ";".join(locations)
        approach_query = ";".join([APPROACH] * len(nodes))

        # Select first/second halves as source/dest sets
        sources_query = ";".join([str(x) for x in range(len(nodes))])
        dests_query = ";".join([str(x) for x in range(len(nodes), len(nodes) * 2)])

        return f"{ENDPOINT}{locations_query}?approaches={approach_query}&sources={sources_query}&destinations={dests_query}&access_token={self.__token}"

    def query_mapbox(self, nodes: list[tuple[Order, Order]]) -> requests.Response:
        # Do only the hard-to-test, external API part
        # Matrix API v1 has a rate limit of 60s, so allow max 1 retry
        req = requests.get(self.__construct_url(nodes))
        if req.status_code == 429:
            time.sleep(60)
            req = requests.get(self.__construct_url(nodes))
            # Barring API change, this should always work. Probably.
        
        return req


class MapboxDistanceFinder(DistanceFinder):
    def __init__(self, requester: MapboxRequester):
        """
        Inits an asymmetric distance matrix maker that uses a Mapbox's Matrix
        API. "Distances" use drive duration in seconds.

        Wraps the actual requester object, handling only the validation.

        Limitations:
        - Does not currently use real-time traffic; only estimated durations.
        - Locations must be fully reachable. Any unroutable pairs will raise
        an exception regardless of if some path still exists.
        - Capped at 12 nodes for now. Extension is doable in 24/12 block
        strides, but out of quantum project scope with so few qubits.

        Parameters
        ----------
        requester : MapboxRequester
            Allows mock API testing without external code.
        """
        self.__requester = requester

    def build_matrix(self, nodes: list[tuple[Order, Order]]) -> np.ndarray:
        """
        Converts TSP nodes into a distance matrix. To create an asymmetric
        matrix, set the start and end coords per "node" to differ.

        Mapbox is subject to extra limitations on input size and data.

        Parameters
        ----------
        nodes: list of (start_order, end_order)
            Unordered list of "TSP cities", each with start and end
            coordinates as orders.

        Returns
        -------
        ndarray
            Asymmetric distance matrix

        Raises
        ------
        ValueError
            Node count is not between 1-12 inclusive.
        RuntimeError
            Node set contains unroutable pairs (e.g. roadless towns, crosses
            the ocean, etc).
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
            # Fetch and extract request to a matrix
            req = self.__requester.query_mapbox(nodes)
            matrix_data = req.json()["durations"]
            distance_matrix = np.array(matrix_data, dtype="float64")
            # NOTE: InvalidInput 422 should be impossible with valid Orders
            # Everything else shouldn't be caught (internal server error)

            # Potentially unroutable coordinate pairs
            if np.isnan(distance_matrix).any():
                # Checking that any reachable path exists in a directed graph
                # is a Hamiltonian Path problem. For now, avoid complex
                # fallback logic and treat as a limitation of road routing.
                raise RuntimeError("Mapbox found unroutable location pairs.")

            return distance_matrix
