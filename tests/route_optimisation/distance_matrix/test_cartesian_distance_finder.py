import pytest
import numpy as np
from pydantic_models import Order, OrderInput
from route_optimisation.distance_matrix.cartesian_distance_finder import CartesianDistanceFinder

# Test invalid inputs
# Test with just ints
# Test with doubles
# Test with mixed datatype


# Order is incorrect
# Orders have mixed inputs

# 1x1 Matrix
# 1x2 Matrix
# 2X1 Matrix
# 2x2 Matrix
# Matrix resulting in zeroes
# Symmetric Test
# Negative Coordinates
# Mixed Coordinates
# Extreme values

# Helper function
def list_to_start_ends(orders : list[Order]) -> list[tuple[Order, Order]]:
    # Converts a list of orders to a list of tuples containing start orders and end orders
    return [(o, o) for o in orders]

@pytest.fixture
def dummy_cdf() -> CartesianDistanceFinder:
    return CartesianDistanceFinder()

@pytest.fixture
def dummy_zeroes_1x1():
    return list_to_start_ends([[0,0,0]])

@pytest.fixture
def dummy_zeroes_2x2():
    return list_to_start_ends([[0,0,0],[0,0,0],
                           [0,0,0],[0,0,0]])

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

def test_matrix_sizes(dummy_cdf : CartesianDistanceFinder, dummy_single_order, dummy_two_orders, dummy_four_orders):
    
    # Test matrix of size 0
    with pytest.raises(ValueError):
        matrix = dummy_cdf.build_matrix(list(tuple()))
    
    # Test matrix of size 1
    matrix = dummy_cdf.build_matrix(dummy_single_order)
    assert np.array_equal(matrix, np.array([[0]]))

    # Test matrix of size 2
    matrix = dummy_cdf.build_matrix(dummy_two_orders)
    assert np.array_equal(matrix, np.array([[0,0],[0,0]]))

    # Test matrix of size 4
    matrix = dummy_cdf.build_matrix(dummy_four_orders)
    assert np.array_equal(matrix, np.array([[0,0,0,0],
                                            [0,0,0,0],
                                            [0,0,0,0],
                                            [0,0,0,0],]))
    
