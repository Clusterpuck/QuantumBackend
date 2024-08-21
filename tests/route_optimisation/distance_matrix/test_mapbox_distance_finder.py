import numpy as np
import pytest
import requests

from pydantic_models import Order
from route_optimisation.distance_matrix.mapbox_distance_finder import (
    MapboxRequester,
    MapboxDistanceFinder,
)


class MockResponse:
    # Ducktypes a requests.Response
    def __init__(self, status_code: int, json: dict):
        self.__status_code = status_code
        self.__json = json

    @property
    def status_code(self):
        return self.__status_code

    def json(self):
        return self.__json


class MockMapboxRequester:
    # Ducktypes actual requester type
    def __init__(self, status_code: int, json: dict):
        self.__response = MockResponse(status_code, json)

    def query_mapbox(self, _: list[tuple[Order, Order]]) -> requests.Response:
        return self.__response


@pytest.fixture
def dummy_orders() -> list[Order]:
    order_data = {
        "order_id": [16, 12, 13, 14, 15, 11, 17, 18],
        "lat": [
            -31.899364,
            -32.010274,
            -32.090316,
            -32.000879,
            -31.900399,
            -32.040650,
            -20,
            -10,
        ],
        "lon": [
            115.801288,
            115.886444,
            115.870573,
            115.920247,
            115.799830,
            115.905166,
            20,
            10,
        ],
        "x": [0, 0, 0, 0, 1, 1, 1, 2],  # Unused fields
        "y": [0, 0, 0, 1, 1, 1, 2, 2],
        "z": [0, 0, 1, 1, 1, 2, 2, 2],
    }

    #  Flip dict[list] to list[dict], then convert to list[Order]
    return [Order(**dict(zip(order_data, t))) for t in zip(*order_data.values())]


@pytest.fixture
def dummy_nodes(dummy_orders: list[Order]) -> list[tuple[Order, Order]]:
    # Create 4 nodes with asymmetric start/ends
    return [(dummy_orders[i], dummy_orders[i + 4]) for i in range(4)]


@pytest.fixture
def expected_url() -> str:
    # The expected URL constructed from dummy_nodes
    endpoint = "https://api.mapbox.com/directions-matrix/v1/mapbox/driving/"

    # Locations should be "lon0,lat0;lon1,lat1;...;lon7,lat7"
    # First half should be starts, latter being end locations
    starts = "115.801288,-31.899364;115.886444,-32.010274;115.870573,-32.090316;115.920247,-32.000879"
    ends = "115.79983,-31.900399;115.905166,-32.04065;20.0,-20.0;10.0,-10.0"
    locations = f"{starts};{ends}"
    # NOTE: Order auto-casts coords to float (.0), due to Pydantic
    # Seems trailing decimal 0s beyond tenths are also dropped

    # Approaches should match input count, and always be curb
    approaches = "?approaches=curb;curb;curb;curb"

    # Sources/destinations should index the first and 2nd half respectively
    sources = "&sources=0;1;2;3"
    dests = "&destinations=4;5;6;7"

    # And finally, the token gets arbitrarily placed last
    token = "&access_token=test_string"

    return f"{endpoint}{locations}{approaches}{sources}{dests}{token}"


@pytest.fixture
def expected_short_url() -> str:
    # Same rules as expected_asym_url, but with the first 2 nodes
    endpoint = "https://api.mapbox.com/directions-matrix/v1/mapbox/driving/"
    starts = "115.801288,-31.899364;115.886444,-32.010274"
    ends = "115.79983,-31.900399;115.905166,-32.04065"
    locations = f"{starts};{ends}"
    approaches = "?approaches=curb;curb"
    sources = "&sources=0;1"
    dests = "&destinations=2;3"
    token = "&access_token=test_string"

    return f"{endpoint}{locations}{approaches}{sources}{dests}{token}"


def test_construct_url(
    dummy_nodes: list[tuple[Order, Order]], expected_url: str, expected_short_url: str
):
    # Check that the final string is correct
    requester = MapboxRequester(token="test_string")
    result = requester._MapboxRequester__construct_url(dummy_nodes)
    assert result == expected_url

    # Check that half of it (2 asym nodes) builds correctly, too
    result = requester._MapboxRequester__construct_url(dummy_nodes[:2])
    assert result == expected_short_url


def test_pre_validation(dummy_nodes: list[tuple[Order, Order]]):
    # Covers all the pre-validation cases before hitting the API
    distance_finder = MapboxDistanceFinder(requester=None)

    # Check that 0x0 is impossible
    with pytest.raises(ValueError):
        distance_finder.build_matrix([])

    # Check that 13x13 is impossible
    with pytest.raises(ValueError):
        distance_finder.build_matrix([dummy_nodes[0]] * 13)

    # Check that 1x1 is strictly a 2D matrix of [[0]]
    result = distance_finder.build_matrix([dummy_nodes[0]])
    assert result.shape == (1, 1)
    assert result[0][0] == 0
    assert np.issubdtype(result.dtype, np.floating)


def test_post_validation(dummy_orders: list[Order]):
    # Covers what happens after the API runs
    # Uses the MockRequester obj to fake a known response

    # Check that 2x2 passes through without errors
    distance_finder = MapboxDistanceFinder(
        requester=MockMapboxRequester(200, {"durations": [[0, 3], [2, 0]]})
    )
    result = distance_finder.build_matrix(dummy_orders)
    expected = np.array([[0, 3], [2, 0]], dtype=np.float64)
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_array_almost_equal(result, expected)

    # Check that 12x12 passes through without errors
    distance_finder = MapboxDistanceFinder(
        requester=MockMapboxRequester(200, {"durations": [[0] * 12] * 12})
    )
    result = distance_finder.build_matrix(dummy_orders)
    expected = np.array([[0] * 12] * 12, dtype=np.float64)
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_array_almost_equal(result, expected)

    # Check that one or more Nones cause immediate rejection (node data error)
    distance_finder = MapboxDistanceFinder(
        requester=MockMapboxRequester(
            200, {"durations": [[0, 3, None], [2, 0, 3], [1.3223, 11, 0]]}
        )
    )
    with pytest.raises(RuntimeError):
        distance_finder.build_matrix(dummy_orders)

    distance_finder = MapboxDistanceFinder(
        requester=MockMapboxRequester(
            200, {"durations": [[0, 3, None], [None, None, None], [1.3223, None, 0]]}
        )
    )
    with pytest.raises(RuntimeError):
        distance_finder.build_matrix(dummy_orders)
