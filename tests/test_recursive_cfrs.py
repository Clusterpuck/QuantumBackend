import numpy as np
import pytest

from pydantic_models import Order, OrderInput
from route_optimisation.recursive_cfrs import RecursiveCFRS
from route_optimisation.distance_matrix.cartesian_distance_finder import (
    CartesianDistanceFinder,
)
from route_optimisation.route_solver.brute_force_solver import BruteForceSolver


# Helper functions
def orders_to_cartesian(
    orders: list[OrderInput],
) -> list[Order]:
    # Borrowed from app for now for fixture set up

    r = 6371  # Radius of the earth
    cartesian_orders = []

    for order in orders:
        r_lat, r_lon = np.deg2rad(order.lat), np.deg2rad(order.long)
        cartesian_orders.append(
            Order(
                order_id=order.order_id,
                lat=order.lat,
                long=order.long,
                x=r * np.cos(r_lat) * np.cos(r_lon),
                y=r * np.cos(r_lat) * np.sin(r_lon),
                z=r * np.sin(r_lat),
            )
        )

    return cartesian_orders


def assert_cluster_trees(
    node1: dict | Order | list, node2: dict | Order | list
) -> None:
    # Compare both cluster tree shapes while recursing
    if isinstance(node1, list):
        assert isinstance(node2, list)
        assert len(node1) == len(node2)

        for i in range(len(node1)):
            assert_cluster_trees(node1[i], node2[i])

    # At lowest objects, compare the data
    else:
        assert not isinstance(node2, list)

        # Convert to dict, if not yet
        item1 = vars(node1) if isinstance(node1, Order) else node1
        item2 = vars(node2) if isinstance(node2, Order) else node2

        # Check properties are equal via ducktype (and floats approx equal)
        assert item1["order_id"] == item2["order_id"]
        assert item1["lat"] == pytest.approx(item2["lat"])
        assert item1["long"] == pytest.approx(item2["long"])
        assert item1["x"] == pytest.approx(item2["x"])
        assert item1["y"] == pytest.approx(item2["y"])
        assert item1["z"] == pytest.approx(item2["z"])
        # Pytest should catch and reveal missing attributes, too


# Fixtures
@pytest.fixture
def dummy_orders() -> list[Order]:
    # Probably would be better to use one calculable by hand, but oh well...
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
        "long": [
            115.801288,
            115.886444,
            115.870573,
            115.920247,
            115.799830,
            115.905166,
            20,
            10,
        ],
    }

    #  Flip dict[list] to list[dict], then convert to list[Order]
    raw_orders = [
        OrderInput(**dict(zip(order_data, t))) for t in zip(*order_data.values())
    ]

    # Add Cartesians and return
    return orders_to_cartesian(raw_orders)


@pytest.fixture
def expected_orders() -> list[list[dict]]:
    # As JSON
    return [
        [
            {
                "order_id": 16,
                "lat": -31.899364,
                "long": 115.801288,
                "x": -2354.2031043853,
                "y": 4869.623654780412,
                "z": -3366.620591053539,
            },
            {
                "order_id": 15,
                "lat": -31.900399,
                "long": 115.79983,
                "x": -2354.052717921549,
                "y": 4869.628806467785,
                "z": -3366.7182965709057,
            },
            {
                "order_id": 14,
                "lat": -32.000879,
                "long": 115.920247,
                "x": -2361.6973871630744,
                "y": 4859.352837709582,
                "z": -3376.19852054214,
            },
            {
                "order_id": 12,
                "lat": -32.010274,
                "long": 115.886444,
                "x": -2358.588355482576,
                "y": 4860.24720649239,
                "z": -3377.0844024342828,
            },
            {
                "order_id": 11,
                "lat": -32.04065,
                "long": 115.905166,
                "x": -2359.39384013173,
                "y": 4857.865073594457,
                "z": -3379.948022366553,
            },
            {
                "order_id": 13,
                "lat": -32.090316,
                "long": 115.870573,
                "x": -2355.1811185798524,
                "y": 4856.650641028967,
                "z": -3384.628110986181,
            },
        ],
        [
            {
                "order_id": 17,
                "lat": -20,
                "long": 20,
                "x": 5625.734573555505,
                "y": 2047.599930656471,
                "z": -2179.0103331278356,
            },
            {
                "order_id": 18,
                "lat": -10,
                "long": 10,
                "x": 6178.890843513511,
                "y": 1089.5051665639178,
                "z": -1106.312539916013,
            },
        ],
    ]


@pytest.fixture
def expected_vrp() -> list[list[dict]]:
    # As JSON, with 2 vehicles
    return [
        [
            {
                "order_id": 16,
                "lat": -31.899364,
                "long": 115.801288,
                "x": -2354.2031043853,
                "y": 4869.623654780412,
                "z": -3366.620591053539,
            },
            {
                "order_id": 15,
                "lat": -31.900399,
                "long": 115.79983,
                "x": -2354.052717921549,
                "y": 4869.628806467785,
                "z": -3366.7182965709057,
            },
            {
                "order_id": 14,
                "lat": -32.000879,
                "long": 115.920247,
                "x": -2361.6973871630744,
                "y": 4859.352837709582,
                "z": -3376.19852054214,
            },
            {
                "order_id": 12,
                "lat": -32.010274,
                "long": 115.886444,
                "x": -2358.588355482576,
                "y": 4860.24720649239,
                "z": -3377.0844024342828,
            },
            {
                "order_id": 11,
                "lat": -32.04065,
                "long": 115.905166,
                "x": -2359.39384013173,
                "y": 4857.865073594457,
                "z": -3379.948022366553,
            },
            {
                "order_id": 13,
                "lat": -32.090316,
                "long": 115.870573,
                "x": -2355.1811185798524,
                "y": 4856.650641028967,
                "z": -3384.628110986181,
            },
        ],
        [
            {
                "order_id": 17,
                "lat": -20,
                "long": 20,
                "x": 5625.734573555505,
                "y": 2047.599930656471,
                "z": -2179.0103331278356,
            },
            {
                "order_id": 18,
                "lat": -10,
                "long": 10,
                "x": 6178.890843513511,
                "y": 1089.5051665639178,
                "z": -1106.312539916013,
            },
        ],
    ]


@pytest.fixture
def expected_tree() -> list[list[dict]]:
    # Matches expected_order, but with the full clustering path
    # Vehicle 1 subdivided [[16, 15], [14, 12, 11], [13]], vehicle 2 [17, 18]
    return [
        [
            [
                {
                    "order_id": 16,
                    "lat": -31.899364,
                    "long": 115.801288,
                    "x": -2354.2031043853,
                    "y": 4869.623654780412,
                    "z": -3366.620591053539,
                },
                {
                    "order_id": 15,
                    "lat": -31.900399,
                    "long": 115.79983,
                    "x": -2354.052717921549,
                    "y": 4869.628806467785,
                    "z": -3366.7182965709057,
                },
            ],
            [
                {
                    "order_id": 14,
                    "lat": -32.000879,
                    "long": 115.920247,
                    "x": -2361.6973871630744,
                    "y": 4859.352837709582,
                    "z": -3376.19852054214,
                },
                {
                    "order_id": 12,
                    "lat": -32.010274,
                    "long": 115.886444,
                    "x": -2358.588355482576,
                    "y": 4860.24720649239,
                    "z": -3377.0844024342828,
                },
                {
                    "order_id": 11,
                    "lat": -32.04065,
                    "long": 115.905166,
                    "x": -2359.39384013173,
                    "y": 4857.865073594457,
                    "z": -3379.948022366553,
                },
            ],
            [
                {
                    "order_id": 13,
                    "lat": -32.090316,
                    "long": 115.870573,
                    "x": -2355.1811185798524,
                    "y": 4856.650641028967,
                    "z": -3384.628110986181,
                }
            ],
        ],
        [
            {
                "order_id": 17,
                "lat": -20,
                "long": 20,
                "x": 5625.734573555505,
                "y": 2047.599930656471,
                "z": -2179.0103331278356,
            },
            {
                "order_id": 18,
                "lat": -10,
                "long": 10,
                "x": 6178.890843513511,
                "y": 1089.5051665639178,
                "z": -1106.312539916013,
            },
        ],
    ]


# Tests
def test_solve_vrp(
    dummy_orders: list[Order],
    expected_vrp: list[list[dict]],
    expected_tree: list[list[Order | list]],
) -> None:
    # NOTE: This test is subject to change, as better data and mocks are added
    # Would also like to mock clustering to a simple halver in the future

    # Set up solver
    dm = CartesianDistanceFinder()
    rs = BruteForceSolver()
    split_threshold = 3
    vrp_solver = RecursiveCFRS(dm, rs, split_threshold)
    # k-means here has been deterministically seeded

    # Check base VRP works
    result_vrp, cluster_tree = vrp_solver.solve_vrp(dummy_orders, 2)
    assert len(result_vrp) == 2
    assert len(cluster_tree) == 2
    assert_cluster_trees(result_vrp, expected_vrp)
    assert_cluster_trees(cluster_tree, expected_tree)

    # Check ordering makes no difference (if deterministic cluster)
    new_ordering = [0, 3, 2, 1, 4, 6, 5, 7]
    dummy_rearranged_orders = [dummy_orders[i] for i in new_ordering]

    result_vrp, cluster_tree = vrp_solver.solve_vrp(dummy_rearranged_orders, 2)
    assert len(result_vrp) == 2
    assert len(cluster_tree) == 2
    assert_cluster_trees(result_vrp, expected_vrp)
    assert_cluster_trees(cluster_tree, expected_tree)

    # Check vehicle count a bit more
    result_vrp, cluster_tree = vrp_solver.solve_vrp(dummy_orders, 1)
    assert len(result_vrp) == 1
    assert len(cluster_tree) == 1
    result_vrp, cluster_tree = vrp_solver.solve_vrp(dummy_orders, 3)
    assert len(result_vrp) == 3
    assert len(cluster_tree) == 3

    # TODO: Check split thresholds 2 and 6 (which doesn't even subdivide)
