import numpy as np
import pytest

from route_optimisation.clusterer.resources.queued_x_means import QueuedXMeans, Cluster


# This should attempt more thorough x-means logic tests than for its clusterer
# Can visualise tests with https://www.desmos.com/3d


@pytest.fixture
def dummy_cluster() -> Cluster:
    # Unrepresentive data/normal, but tests projection + covariance etc
    data = np.array(
        [
            [-1, 0, 1],  # To [1.4142, 0.0]
            [3, 2, 1],  # To [-1.4142, 0.0]
            [1, 1, 1],  # To [0.0, 0.0]
            [-4, 2, 2],  # To [4.2426, 2.4495]
            [-4, 3, 2],  # To [4.2426, 3.2660]
        ]
    )
    indices = np.array(range(5))
    center = np.array([1, 1, 1])
    # New x basis = [-0.70711, 0.0, 0.70711]
    # New y basis = [-0.40825, 0.81650, -0.40825]
    # Notice x basis will always be vertically flat

    # Free params = 2d * (2d + 3) / 2 = 5.0
    # Covariance = [[6.4, 3.6373], [3.6373, 2.5333]]
    # TODO: Log-likelihood = ?
    # TODO: BIC = ?

    return Cluster(data, indices, center)

@pytest.fixture
def dummy_1d_cluster() -> Cluster:
    # Normal ends up squashing data into a line along x-axis, where y=0, z=1
    data = np.array(
        [
            [-1, 0, 1],  # To [1.0, 0.0]
            [3, 0, 4],  # To [-3.0, 0.0]
            [1, 0, 2],  # To [-1.0, 0.0]
            [-4, 0, -98],  # To [4.0, 0.0]
            [-4, 0, 98],  # To [4.0, 0.0]
        ]
    )
    indices = np.array(range(5))
    center = np.array([0, 0, 1])  # xy plane, where z=1

    # df and cov should succeed, but likelihood and bic should fail

    return Cluster(data, indices, center)

@pytest.fixture
def dummy_0d_cluster() -> Cluster:
    # Normal ends up squashing data into single point
    data = np.array(
        [
            [20, 0, 1],  # To [20.0, 0.0]
            [20, 0, 4],  # Ditto
            [20, 0, 2],  # Ditto
            [20, 0, -98],  # Ditto
            [20, 0, 98],  # Ditto
        ]
    )
    indices = np.array(range(5))
    center = np.array([0, 0, 1])  # xy plane, where z=1

    # df and cov should succeed, but likelihood and bic should fail

    return Cluster(data, indices, center)

# TODO: Add 0,0,0 and 0,1,0 center to check behaviour


def test_cluster(dummy_cluster: Cluster) -> None:
    pass
    # result = dummy_cluster.fit()
    # assert bic info

    # Test the 1d and 0d known bic failures
    # Test the single data point known bic failure


# TODO: X-means test cases
# 0,0,0 normal should be caught in some way, probably null cov and bics
# points should be validated for 3d initial input (not implemented yet)
# when fit is called... ?
# - Test 3 known "good" clusters with k init 3 each end
# - Test 5 known clusters with k init 2 to check it splits
# - Test on some randomish data for regression sake
