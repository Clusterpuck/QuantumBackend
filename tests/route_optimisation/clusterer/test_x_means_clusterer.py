import numpy as np
import pytest

from pydantic_models import Order
from route_optimisation.clusterer.x_means_clusterer import XMeansClusterer


@pytest.fixture
def dummy_orders() -> list[Order]:
    order_data = {
        "order_id": [16, 12, 13, 14, 15, 11, 17, 18],
        "lat": [0, 0, 0, 0, 0, 0, 0, 0],  # Won't need these
        "lon": [0, 0, 0, 0, 0, 0, 0, 0],
        "x": [-30, -20, 20, 20, -20, 10, 20, 58],
        "y": [-30, -40, 30, 20, -20, 40, 40, 35],
        "z": [-30, -30, 10, 20, -20, 20, 10, 45],
    }
    # With k=2, [0, 0, 1, 1, 0, 1, 1, 1]
    # With k=3, [0, 0, 1, 1, 0, 1, 1, 2]

    # TODO: change data to 3D gaussian, then add another set with near flats

    #  Flip dict[list] to list[dict], then convert to list[Order]
    orders = [Order(**dict(zip(order_data, t))) for t in zip(*order_data.values())]

    # Add Cartesians and return
    return orders


def test_cluster(dummy_orders: list[Order]) -> None:
    # Check k_max=2, k_init=2
    clusterer = XMeansClusterer(2, k_init=2)
    np.testing.assert_array_equal(
        clusterer.cluster(dummy_orders), np.array([1, 1, 0, 0, 1, 0, 0, 0])
    )
    # Exact opposite of k-means, so it seems to work

    # Check k_max=3, k_init=3
    clusterer = XMeansClusterer(3, k_init=3)
    np.testing.assert_array_equal(
        clusterer.cluster(dummy_orders), np.array([0, 0, 1, 1, 0, 1, 1, 2])
    )

    # Check k_max=6, k_init=1
    # In other words, k grows deterministically as expected
    clusterer = XMeansClusterer(6, k_init=1)
    np.testing.assert_array_equal(
        clusterer.cluster(dummy_orders), np.array([0, 0, 1, 1, 0, 1, 1, 2])
    )

    # Check exact number of points succeeds (known previous bug)
    XMeansClusterer(2, k_init=2).cluster(dummy_orders[0:2])

    # Check shuffled input produces shuffled output
    clusterer = XMeansClusterer(3)
    new_ordering = [7, 3, 2, 1, 5, 6, 4, 0]
    dummy_rearranged_orders = [dummy_orders[i] for i in new_ordering]
    np.testing.assert_array_equal(
        clusterer.cluster(dummy_rearranged_orders), np.array([2, 0, 0, 1, 0, 0, 1, 1])
    )
    # Don't know the exact rules for clustpy's labelling, but assume that
    # changes might mean an error. This can be more rigourously checked with
    # some linear mapping tracker.


def test_failures(dummy_orders: list[Order]) -> None:
    # Check that constructor k validation can be tripped
    with pytest.raises(ValueError):
        XMeansClusterer(0)
    with pytest.raises(ValueError):
        XMeansClusterer(1, k_init=2)
    with pytest.raises(ValueError):
        XMeansClusterer(1, k_init=0)

    # Check too few points fails
    with pytest.raises(ValueError):
        XMeansClusterer(2, k_init=2).cluster([dummy_orders[0]])
    with pytest.raises(ValueError):
        XMeansClusterer(6, k_init=6).cluster(dummy_orders[0:2])

    # Check duplicate clusters fail
    with pytest.raises(RuntimeError):
        XMeansClusterer(2, k_init=2).cluster([dummy_orders[0]] * 2)
    with pytest.raises(RuntimeError):
        XMeansClusterer(3, k_init=2).cluster([dummy_orders[0]] * 3)
    with pytest.raises(RuntimeError):
        XMeansClusterer(3, k_init=3).cluster([dummy_orders[0]] * 3)
