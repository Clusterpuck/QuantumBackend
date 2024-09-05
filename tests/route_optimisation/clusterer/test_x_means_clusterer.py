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
        "x": [-30, -40, 50, 50, -40, 40, 40, 38],
        "y": [6371] * 8,
        "z": [-30, -20, 20, 30, -30, 30, 40, 45],
    }

    # NOTE: Data has to be at least somewhat realistic to split well. It will
    # run, but it might decide to stop early if the cluster scores poorly.

    #  Flip dict[list] to list[dict], then convert to list[Order]
    orders = [Order(**dict(zip(order_data, t))) for t in zip(*order_data.values())]

    # Add Cartesians and return
    return orders


def test_cluster(dummy_orders: list[Order]) -> None:
    # Assume thorough testing is done for underlying x-means
    # So just do basic tests for the basic clusterer layer and its determinism

    # Check that k_max=k_init works like k-means
    clusterer = XMeansClusterer(2, k_init=2)
    np.testing.assert_array_equal(
        clusterer.cluster(dummy_orders), np.array([1, 1, 0, 0, 1, 0, 0, 0])
    )
    clusterer = XMeansClusterer(3, k_init=3)
    np.testing.assert_array_equal(
        clusterer.cluster(dummy_orders), np.array([0, 0, 1, 1, 0, 1, 2, 2])
    )

    # Check that k can grow somewhat even on small data if the cluster is ok
    clusterer = XMeansClusterer(6, k_init=1)
    np.testing.assert_array_equal(
        clusterer.cluster(dummy_orders), np.array([1, 1, 0, 0, 1, 0, 0, 0])
    )

    # Check exact number of points succeeds (known previous bug)
    XMeansClusterer(2, k_init=2).cluster(dummy_orders[0:2])

    # Check shuffled input produces shuffled output
    clusterer = XMeansClusterer(3, k_init=3)
    new_ordering = [7, 3, 2, 1, 5, 6, 4, 0]
    dummy_rearranged_orders = [dummy_orders[i] for i in new_ordering]
    np.testing.assert_array_equal(
        clusterer.cluster(dummy_rearranged_orders), np.array([2, 1, 1, 0, 1, 2, 0, 0])
    )
    # Again, swapped labelling in the test fails suggest possible logic change


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

    # NOTE: Removed duplicate check, since we currently support unbounded min
    # Re-add those tests if we ever adapt x-means to route subclustering etc
