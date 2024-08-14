import numpy as np
import pytest

from pydantic_models import Order
from route_optimisation.clusterer.k_means_clusterer import KMeansClusterer


# Fixtures
@pytest.fixture
def expected_minimal_fallback_split() -> np.ndarray:
    # Input: [0 0 1], k=3
    # Expected behaviour:
    # Builds missing labels [2]
    # [2 0 1], Cluster 0 has the most points, so add the missing label
    return np.array([2, 0, 1])


@pytest.fixture
def expected_tied_fallback_split() -> np.ndarray:
    # Input: [1 1 1 1 0 0 0 0], k=3
    # Expected behaviour:
    # Builds missing labels [2]
    # [2 2 1 1 0 0 0 0], Clusters 1 and 0 are tied, so pick by lowest label
    return np.array([1, 1, 1, 1, 2, 2, 0, 0])


@pytest.fixture
def expected_half_fallback_split() -> np.ndarray:
    # Input: [0 0 1 1 1], k=3
    # Expected behaviour:
    # Builds missing labels [2]
    # [0 0 2 1 1], Halving should be floored division
    return np.array([0, 0, 2, 1, 1])


@pytest.fixture
def expected_many_fallback_split() -> np.ndarray:
    # Input: [0 0 0 1 1 3 4 7 7 7 7 7 7 7], k=9
    # Expected behaviour:
    # Builds missing labels [2, 5, 6, 8]
    # Iteratively writes each onto half the next largest cluster
    # [0 0 0 1 1 3 4 2 2 2 7 7 7 7], 7 appears x7, writes 2s
    # [0 0 0 1 1 3 4 2 2 2 5 5 7 7], 7 appears x4, writes 5s
    # [6 0 0 1 1 3 4 2 2 2 5 5 7 7], 0/2 tied x3 but picks lower 0, writes 6
    # [6 0 0 1 1 3 4 8 2 2 5 5 7 7], 2 left as x3, so write 8
    return np.array([6, 0, 0, 1, 1, 3, 4, 8, 2, 2, 5, 5, 7, 7])


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

    #  Flip dict[list] to list[dict], then convert to list[Order]
    orders = [Order(**dict(zip(order_data, t))) for t in zip(*order_data.values())]

    # Add Cartesians and return
    return orders


def test_fallback_split(
    expected_minimal_fallback_split: np.ndarray,
    expected_tied_fallback_split: np.ndarray,
    expected_half_fallback_split: np.ndarray,
    expected_many_fallback_split: np.ndarray,
) -> None:
    clusterer = KMeansClusterer(3)  # This k doesn't matter here

    # Check minimal case
    labels = np.array([0, 0, 1])
    k = 3
    np.testing.assert_array_equal(
        clusterer._KMeansClusterer__fallback_split(labels, k),
        expected_minimal_fallback_split,
    )

    # Check tied case
    labels = np.array([1, 1, 1, 1, 2, 2, 0, 0])
    k = 3
    np.testing.assert_array_equal(
        clusterer._KMeansClusterer__fallback_split(labels, k),
        expected_tied_fallback_split,
    )

    # Check odd halving case
    labels = np.array([0, 0, 1, 1, 1])
    k = 3
    np.testing.assert_array_equal(
        clusterer._KMeansClusterer__fallback_split(labels, k),
        expected_half_fallback_split,
    )

    # Check extended iterative case on bigger k
    labels = np.array([0, 0, 0, 1, 1, 3, 4, 7, 7, 7, 7, 7, 7, 7])
    k = 9
    np.testing.assert_array_equal(
        clusterer._KMeansClusterer__fallback_split(labels, k),
        expected_many_fallback_split,
    )

    # NOTE: Not currently expecting any validation logic from this helper


def test_cluster(dummy_orders: list[Order]) -> None:
    # Check k=2
    clusterer = KMeansClusterer(2)
    np.testing.assert_array_equal(
        clusterer.cluster(dummy_orders), np.array([0, 0, 1, 1, 0, 1, 1, 1])
    )

    # Check k=3
    clusterer = KMeansClusterer(3)
    np.testing.assert_array_equal(
        clusterer.cluster(dummy_orders), np.array([0, 0, 1, 1, 0, 1, 1, 2])
    )

    # Check shuffled input produces shuffled output
    clusterer = KMeansClusterer(3)
    new_ordering = [7, 3, 2, 1, 5, 6, 4, 0]
    dummy_rearranged_orders = [dummy_orders[i] for i in new_ordering]
    np.testing.assert_array_equal(
        clusterer.cluster(dummy_rearranged_orders), np.array([2, 0, 0, 1, 0, 0, 1, 1])
    )
    # Don't know the exact rules for scikit's labelling, but assume that
    # changes might mean an error. This can be more rigourously checked with
    # some linear mapping tracker.


def test_configs(dummy_orders: list[Order]) -> None:
    # Check that constructor literal validation can be tripped
    with pytest.raises(ValueError):
        KMeansClusterer(2, allow_less_data=True, duplicate_clusters="text")

    # Check too few points can be permitted
    result = KMeansClusterer(
        2, allow_less_data=True, duplicate_clusters="allow"
    ).cluster([dummy_orders[0]])
    assert len(np.unique(result)) == 1

    result = KMeansClusterer(
        6, allow_less_data=True, duplicate_clusters="raise"
    ).cluster(dummy_orders[0:2])
    assert len(np.unique(result)) == 2

    # Check too few points can be explicitly disallowed
    with pytest.raises(ValueError):
        KMeansClusterer(2, allow_less_data=False, duplicate_clusters="allow").cluster(
            [dummy_orders[0]]
        )
    with pytest.raises(ValueError):
        KMeansClusterer(6, allow_less_data=False, duplicate_clusters="raise").cluster(
            dummy_orders[0:2]
        )

    # Check duplicate clusters can be permitted (with or without less data)
    result = KMeansClusterer(
        2, allow_less_data=True, duplicate_clusters="allow"
    ).cluster([dummy_orders[0]] * 2)
    assert len(np.unique(result)) == 1

    result = KMeansClusterer(
        4, allow_less_data=True, duplicate_clusters="allow"
    ).cluster([dummy_orders[0]] * 3)
    assert len(np.unique(result)) == 1

    # Check duplicate clusters can be explicitly disallowed
    with pytest.raises(RuntimeError):
        KMeansClusterer(2, allow_less_data=True, duplicate_clusters="raise").cluster(
            [dummy_orders[0]] * 2
        )

    with pytest.raises(RuntimeError):
        KMeansClusterer(4, allow_less_data=True, duplicate_clusters="raise").cluster(
            [dummy_orders[0]] * 3
        )

    # Check that forced split works (with or without less data)
    result = KMeansClusterer(
        3, allow_less_data=True, duplicate_clusters="split"
    ).cluster([dummy_orders[0]] * 3)
    assert result.size == 3 and len(np.unique(result)) == 3

    result = KMeansClusterer(
        4, allow_less_data=True, duplicate_clusters="split"
    ).cluster([dummy_orders[0]] * 3)
    assert result.size == 3 and len(np.unique(result)) == 3
    # Allowing less data reduces k, so it should still be all unique

    result = KMeansClusterer(
        3, allow_less_data=True, duplicate_clusters="split"
    ).cluster([dummy_orders[0]] * 8)
    assert result.size == 8 and len(np.unique(result)) == 3
