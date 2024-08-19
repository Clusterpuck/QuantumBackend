import math
import pytest
import numpy as np
from pydantic_models import Order, OrderInput
from route_optimisation.distance_matrix.cartesian_distance_finder import CartesianDistanceFinder

# Helper function
def list_to_start_ends(orders : list[Order]) -> list[tuple[Order, Order]]:
    # Converts a list of orders to a list of tuples containing start orders and end orders
    return [(o, o) for o in orders]

# Fixtures
@pytest.fixture
def dummy_cdf() -> CartesianDistanceFinder:
    return CartesianDistanceFinder()

@pytest.fixture
def dummy_single_order() -> list[Order]:
    order_data = {
        "order_id": [0],
        "lat": [0],
        "lon": [0],
        "x": [0],
        "y": [3],
        "z": [7.7],
    }
    #  Flip dict[list] to list[dict], then convert to list[Order]
    orders = [Order(**dict(zip(order_data, t))) for t in zip(*order_data.values())]
    matrix = list_to_start_ends(orders)
    return matrix

@pytest.fixture
def dummy_two_orders() -> list[Order]:
    order_data = {
        "order_id": [0, 1],
        "lat": [0, 0],  # Won't need these
        "lon": [0, 0],
        "x": [0, 0],
        "y": [0, 0],
        "z": [0, 0],
    }
    #  Flip dict[list] to list[dict], then convert to list[Order]
    orders = [Order(**dict(zip(order_data, t))) for t in zip(*order_data.values())]
    matrix = list_to_start_ends(orders)
    return matrix

@pytest.fixture
def dummy_four_orders() -> list[Order]:
    order_data = {
        "order_id": [16, 12, 13, 14],
        "lat": [0, 0, 0, 0],
        "lon": [0, 0, 0, 0],
        "x": [0, 0, 0, 0],
        "y": [0, 0, 0, 0],
        "z": [0, 0, 0, 0]
    }
    #  Flip dict[list] to list[dict], then convert to list[Order]
    orders = [Order(**dict(zip(order_data, t))) for t in zip(*order_data.values())]
    matrix = list_to_start_ends(orders)
    return matrix

@pytest.fixture
def dummy_mixed_types() -> list[Order]:
    order_data = {
        "order_id": [21, 12],
        "lat": [0, 0],
        "lon": [0, 0],
        "x": [0, 0.42],
        "y": [51, 42],
        "z": [12.52, 63.2]
    }
    #  Flip dict[list] to list[dict], then convert to list[Order]
    orders = [Order(**dict(zip(order_data, t))) for t in zip(*order_data.values())]
    matrix = list_to_start_ends(orders)
    return matrix

@pytest.fixture
def normal_orders() -> list[Order]:
    order1 = Order(order_id=1,
              lat=0,
              lon=0,
              x=0,
              y=0,
              z=0)
    order2 = Order(order_id=2,
              lat=0,
              lon=0,
              x=5,
              y=5,
              z=5)
    orders = [order1, order2]
    matrix = list_to_start_ends(orders)
    return matrix

@pytest.fixture
def negative_orders() -> list[Order]:
    order1 = Order(order_id=1,
              lat=0,
              lon=0,
              x=5,
              y=5,
              z=5)
    order2 = Order(order_id=2,
              lat=0,
              lon=0,
              x=-5,
              y=-5,
              z=-5)
    orders = [order1, order2]
    matrix = list_to_start_ends(orders)
    return matrix

@pytest.fixture
def extreme_orders() -> list[Order]:
    order1 = Order(order_id=1,
              lat=0,
              lon=0,
              x=0,
              y=0,
              z=0)
    order2 = Order(order_id=2,
              lat=0,
              lon=0,
              x=1e-10,
              y=1e-10,
              z=1e-10)
    orders = [order1, order2]
    matrix = list_to_start_ends(orders)
    return matrix

@pytest.fixture
def dummy_invalid_tuple() -> list[Order]:
    order1 = Order(order_id=1,
              lat=0,
              lon=0,
              x=0,
              y=0,
              z=0)
    order2 = Order(order_id=2,
              lat=0,
              lon=0,
              x=5,
              y=5,
              z=5)
    orders = [(order1, order2)]
    matrix = list_to_start_ends(orders)
    return matrix

@pytest.fixture
def dummy_invalid_no_order() -> list[Order]:
    order1 = (0, 0, 0)
    order2 = (5, 5, 5)
    orders = [order1, order2]
    matrix = list_to_start_ends(orders)
    return matrix

@pytest.fixture
def expected_mixed_types() -> list[list]:
    distance = math.dist((0, 51, 12.52), (0.42, 42, 63.2))
    return [[0, distance],[distance, 0]]

@pytest.fixture
def expected_normal_orders() -> list[list]:
    distance = math.dist((0, 0, 0), (5, 5, 5))
    return [[0, distance],[distance, 0]]

@pytest.fixture
def expected_negative_orders() -> list[list]:
    distance = math.dist((-5, -5, -5), (5, 5, 5))
    return [[0, distance],[distance, 0]]

@pytest.fixture
def expected_extreme_orders() -> list[list]:
    distance = math.dist((0, 0, 0), (1e-10, 1e-10, 1e-10))
    return [[0, distance],[distance, 0]]

# Tests
def test_matrix_size(dummy_cdf : CartesianDistanceFinder, 
                      dummy_single_order : list[Order], 
                      dummy_two_orders : list[Order] , 
                      dummy_four_orders : list[Order]) -> None:
    """
    Test with matrices of different sizes
    Currently testing:
        Size 0x0 (Invalid)
        Size 1x1 (Valid)
        Size 2x2 (Valid)
        Size 4x4 (Valid)
    """

    # Test matrix of size 0x0, should raise a value error
    with pytest.raises(ValueError):
        matrix = dummy_cdf.build_matrix(list(tuple()))
    
    # Test matrix of size 1x1
    matrix = dummy_cdf.build_matrix(dummy_single_order)
    assert np.array_equal(matrix, np.array([[0]]))

    # Test matrix of size 2x2
    matrix = dummy_cdf.build_matrix(dummy_two_orders)
    assert np.array_equal(matrix, np.array([[0,0],[0,0]]))

    # Test matrix of size 4x4
    matrix = dummy_cdf.build_matrix(dummy_four_orders)
    assert np.array_equal(matrix, np.array([[0,0,0,0],
                                            [0,0,0,0],
                                            [0,0,0,0],
                                            [0,0,0,0],]))
    
def test_input_values(dummy_cdf : CartesianDistanceFinder,
                      dummy_mixed_types : list[Order],
                      expected_mixed_types,
                      expected_normal_orders,
                      normal_orders,
                      dummy_invalid_tuple,
                      dummy_invalid_no_order) -> None:
    matrix = dummy_cdf.build_matrix(dummy_mixed_types)
    assert np.array_equal(matrix, expected_mixed_types)

    matrix = dummy_cdf.build_matrix(normal_orders)
    assert np.array_equal(matrix, expected_normal_orders)

    with pytest.raises(TypeError):
        matrix = dummy_cdf.build_matrix(dummy_invalid_tuple)

    with pytest.raises(TypeError):
        matrix = dummy_cdf.build_matrix(dummy_invalid_no_order)

def test_matrix_values(dummy_cdf : CartesianDistanceFinder,
                       negative_orders,
                       expected_negative_orders,
                       extreme_orders,
                       expected_extreme_orders):
    matrix = dummy_cdf.build_matrix(negative_orders)
    assert np.array_equal(matrix, expected_negative_orders)

    matrix = dummy_cdf.build_matrix(extreme_orders)
    assert np.array_equal(matrix, expected_extreme_orders)

def test_matrix_dimensions(dummy_cdf : CartesianDistanceFinder):
    pass