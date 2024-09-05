import numpy as np
import pytest

from route_optimisation.clusterer.resources.geo_x_means import GeoXMeans, GeoCluster


# This should attempt more thorough x-means logic tests than for its clusterer
# Can visualise tests with https://www.desmos.com/3d

ABS_TOL = 0.001


@pytest.fixture
def dummy_cluster() -> GeoCluster:
    # Basic test of projection and half-decent distribution
    data = np.array(
        [
            [-1, 0, -1],  # To [0.0, 0.816]
            [3, 2, 3],  # To [0.0, -0.816]
            [1, 2, -3],  # To [2.828, 2.449]
            [-2, 2, -2],  # To [0.0, 3.266]
            [-4, 3, 2],  # To [-4.243, 3.266]
        ]
    )
    indices = np.array(range(5))
    center = np.array([1, 1, 1])
    # New x basis = [0.707, 0.0, -0.707]
    # New y basis = [-0.408, 0.816, -0.408]

    # Free params = 2d * (2d + 3) / 2 = 5.0
    # Covariance = [[6.4, -1.097], [-1.097, 3.133]]
    # Log-likelihood = -73.246
    # BIC = 154.538

    return GeoCluster(data, indices, center)


@pytest.fixture
def dummy_distant_normal_cluster() -> GeoCluster:
    # Tests that distant normal of same direction gives the same as regular
    data = np.array(
        [
            [-1, 0, -1],  # To [0.0, 0.816]
            [3, 2, 3],  # To [0.0, -0.816]
            [1, 2, -3],  # To [2.828, 2.449]
            [-2, 2, -2],  # To [0.0, 3.266]
            [-4, 3, 2],  # To [-4.243, 3.266]
        ]
    )
    indices = np.array(range(5))
    center = np.array([1000, 1000, 1000])
    # New x basis = [0.707, 0.0, -0.707]
    # New y basis = [-0.408, 0.816, -0.408]

    # Free params = 2d * (2d + 3) / 2 = 5.0
    # Covariance = [[6.4, -1.097], [-1.097, 3.133]]
    # Log-likelihood = -73.246
    # BIC = 154.538

    return GeoCluster(data, indices, center)


@pytest.fixture
def dummy_vertical_cluster() -> GeoCluster:
    # Edge case vertical normal should still succeed
    data = np.array(
        [
            [-1, 0, 1],  # To [-1.0, 1.0]
            [3, 2, 1],  # To [-3.0, 1.0]
            [1, 1, 1],  # To [1.0, 1.0]
            [-4, 2, 2],  # To [-4.0, 2.0]
            [-4, 3, 2],  # To [-4.0, 2.0]
        ]
    )
    indices = np.array(range(5))
    center = np.array([0, 1, 0])
    # New x basis = [1.0, 0.0, 0.0]
    # New y basis = [0.0, 0.0, 1.0]
    # Notice x basis will always be vertically flat

    # Free params = 2d * (2d + 3) / 2 = 5.0
    # Covariance = [[9.5, -1.5], [-1.5, 0.3]]
    # Log-likelihood = -73.246
    # BIC = 154.538

    return GeoCluster(data, indices, center)


@pytest.fixture
def dummy_thin_cluster() -> GeoCluster:
    # Overly thin data should just fail
    data = np.array(
        [
            [-1, 0, 1],  # To [1.0, 0.0]
            [3, 0.0001, 4],  # To [-3.0, 0.0001]
            [1, 0, 2],  # To [-1.0, 0.0]
            [-4, 0, -98],  # To [4.0, 0.0]
            [-4, 0, 98],  # To [4.0, 0.0]
        ]
    )
    indices = np.array(range(2))
    center = np.array([0, 0, 1])  # xy plane, where z=1
    # New x basis = [1.0, 0.0, 0.0]
    # New y basis = [0.0, 1.0, 0.0]

    # Free params = 2d * (2d + 3) / 2 = 5.0
    # Covariance = [[9.5, 0.000], [0.000, 0.000]], actually slightly more
    # Log-likelihood = None
    # BIC = None

    return GeoCluster(data, indices, center)


@pytest.fixture
def dummy_1d_cluster() -> GeoCluster:
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
    # New x basis = [1.0, 0.0, 0.0]
    # New y basis = [0.0, 1.0, 0.0]

    # Free params = 2d * (2d + 3) / 2 = 5.0
    # Covariance = [[9.5, 0.0], [0.0, 0.0]]
    # Log-likelihood = None
    # BIC = None

    return GeoCluster(data, indices, center)


@pytest.fixture
def dummy_0d_cluster() -> GeoCluster:
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
    # New x basis = [1.0, 0.0, 0.0]
    # New y basis = [0.0, 1.0, 0.0]

    # Free params = 2d * (2d + 3) / 2 = 5.0
    # Covariance = [[0.0, 0.0], [0.0, 0.0]]
    # Log-likelihood = None
    # BIC = None

    return GeoCluster(data, indices, center)


@pytest.fixture
def dummy_non_invertible_cluster() -> GeoCluster:
    # Singular covariance from degenerate data is treated like regular fails
    data = np.array(
        [
            [1, 1, 0],  # To [1.0, 1.0]
            [-1, -1, 0],  # To [-1.0, -1.0]
        ]
    )
    indices = np.array(range(2))
    center = np.array([0, 0, 1])  # xy plane, where z=1
    # New x basis = [1.0, 0.0, 0.0]
    # New y basis = [0.0, 1.0, 0.0]

    # Free params = 2d * (2d + 3) / 2 = 5.0
    # Covariance = [[2.0, 2.0], [2.0, 2.0]]
    # Log-likelihood = None
    # BIC = None

    return GeoCluster(data, indices, center)


@pytest.fixture
def dummy_nan_cluster() -> GeoCluster:
    # NaN covariance can occur with insufficient data
    data = np.array(
        [
            [1, 1, 0],  # To [1.0, 1.0]
        ]
    )
    indices = np.array(range(2))
    center = np.array([0, 0, 1])  # xy plane, where z=1
    # New x basis = [1.0, 0.0, 0.0]
    # New y basis = [0.0, 1.0, 0.0]

    # Free params = 2d * (2d + 3) / 2 = 5.0
    # Covariance = [[nan, nan], [nan, nan]]
    # Log-likelihood = None
    # BIC = None

    return GeoCluster(data, indices, center)


def test_cluster(
    dummy_cluster: GeoCluster,
    dummy_distant_normal_cluster: GeoCluster,
    dummy_vertical_cluster: GeoCluster,
) -> None:
    # Check indicies passed through
    np.testing.assert_array_equal(dummy_cluster.indices, np.array([0, 1, 2, 3, 4]))

    # Check that a regular cluster works fine
    assert dummy_cluster.df == pytest.approx(5.0, abs=ABS_TOL)
    np.testing.assert_allclose(
        dummy_cluster.cov, np.array([[6.4, -1.097], [-1.097, 3.133]]), atol=ABS_TOL
    )
    assert dummy_cluster.log_likelihood == pytest.approx(-23.155, abs=ABS_TOL)
    assert dummy_cluster.bic == pytest.approx(54.357, abs=ABS_TOL)

    # Check that a longer plane normal changes nothing
    # Notice that the centroid loses its "height", so the BIC is unchanged
    assert dummy_distant_normal_cluster.df == pytest.approx(
        dummy_cluster.df, abs=ABS_TOL
    )
    np.testing.assert_allclose(
        dummy_distant_normal_cluster.cov, dummy_cluster.cov, atol=ABS_TOL
    )
    assert dummy_distant_normal_cluster.log_likelihood == pytest.approx(
        dummy_cluster.log_likelihood, abs=ABS_TOL
    )
    assert dummy_distant_normal_cluster.bic == pytest.approx(
        dummy_cluster.bic, abs=ABS_TOL
    )

    # Check edge case projection to y=0 plane still functions
    assert dummy_vertical_cluster.df == pytest.approx(5.0, abs=ABS_TOL)
    np.testing.assert_allclose(
        dummy_vertical_cluster.cov, np.array([[9.5, -1.5], [-1.5, 0.3]]), atol=ABS_TOL
    )
    assert dummy_vertical_cluster.log_likelihood == pytest.approx(-73.246, abs=ABS_TOL)
    assert dummy_vertical_cluster.bic == pytest.approx(154.538, abs=ABS_TOL)


def test_bad_clusters(
    dummy_thin_cluster: GeoCluster,
    dummy_1d_cluster: GeoCluster,
    dummy_0d_cluster: GeoCluster,
    dummy_non_invertible_cluster: GeoCluster,
    dummy_nan_cluster: GeoCluster,
) -> None:
    # This should succeed, but their log-likelihood and BIC will fail

    # Check cluster with thin-ish distribution (a very low cov eigenvalue)
    assert dummy_thin_cluster.df == pytest.approx(5.0, abs=ABS_TOL)
    np.testing.assert_allclose(
        dummy_thin_cluster.cov, np.array([[9.5, 0.0], [0.0, 0.0]]), atol=ABS_TOL
    )
    assert dummy_thin_cluster.log_likelihood is None
    assert dummy_thin_cluster.bic is None

    # Check cluster with 1 dimension of distribution
    assert dummy_1d_cluster.df == pytest.approx(5.0, abs=ABS_TOL)
    np.testing.assert_allclose(
        dummy_1d_cluster.cov, np.array([[9.5, 0.0], [0.0, 0.0]]), atol=ABS_TOL
    )
    assert dummy_1d_cluster.log_likelihood is None
    assert dummy_1d_cluster.bic is None

    # Check cluster with 0 dimensions of distribution (stacked points)
    assert dummy_0d_cluster.df == pytest.approx(5.0, abs=ABS_TOL)
    np.testing.assert_allclose(
        dummy_0d_cluster.cov, np.array([[0.0, 0.0], [0.0, 0.0]]), atol=ABS_TOL
    )
    assert dummy_0d_cluster.log_likelihood is None
    assert dummy_0d_cluster.bic is None

    # Check that it survives a LinAlgError, using a known triggering cluster
    assert dummy_non_invertible_cluster.df == pytest.approx(5.0, abs=ABS_TOL)
    np.testing.assert_allclose(
        dummy_non_invertible_cluster.cov,
        np.array([[2.0, 2.0], [2.0, 2.0]]),
        atol=ABS_TOL,
    )
    assert dummy_non_invertible_cluster.log_likelihood is None
    assert dummy_non_invertible_cluster.bic is None

    # Check that it survives a NaN covariance matrix
    assert dummy_nan_cluster.df == pytest.approx(5.0, abs=ABS_TOL)
    np.testing.assert_array_equal(
        dummy_nan_cluster.cov, np.array([[np.nan, np.nan], [np.nan, np.nan]])
    )
    assert dummy_nan_cluster.log_likelihood is None
    assert dummy_nan_cluster.bic is None


# def test_x_means() -> None:


# TODO: X-means test cases
# 0,0,0 normal should be caught in some way, probably null cov and bics
# points should be validated for 3d initial input (not implemented yet)
# when fit is called... ?
# - Test 3 known "good" clusters with k init 3 each end
# - Test 5 known clusters with k init 2 to check it splits
# - Test on some randomish data for regression sake

# NOTE: BIC can definitely seem better on semi-thin and small clusters, but
# that is often because the points are closer and the centroid is great. You
# can definitely see it get worse with a bad centroids or data distribution.

# TODO: We know single data points are handled, so can add that check later
# We also know that 0,0,0 normal is handled fine at least for now
