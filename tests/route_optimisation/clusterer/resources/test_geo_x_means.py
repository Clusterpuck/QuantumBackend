import numpy as np
import pytest

from route_optimisation.clusterer.resources.geo_x_means import GeoXMeans, GeoCluster


# This should attempt somewhat thorough x-means logic tests
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

    # Doubled params = 4
    # Covariance = [[6.4, -1.097], [-1.097, 3.133]]
    # Log-likelihood = -23.155
    # BIC = 52.748

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

    # Doubled params = 4
    # Covariance = [[6.4, -1.097], [-1.097, 3.133]]
    # Log-likelihood = -23.155
    # BIC = 52.748

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

    # Doubled params = 4
    # Covariance = [[9.5, -1.5], [-1.5, 0.3]]
    # Log-likelihood = -73.246
    # BIC = 152.929

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
    indices = np.array(range(5))
    center = np.array([0, 0, 1])  # xy plane, where z=1
    # New x basis = [1.0, 0.0, 0.0]
    # New y basis = [0.0, 1.0, 0.0]

    # Doubled params = 4
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

    # Doubled params = 4
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

    # Doubled params = 4
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

    # Doubled params = 4
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
    indices = np.array(range(1))
    center = np.array([0, 0, 1])  # xy plane, where z=1
    # New x basis = [1.0, 0.0, 0.0]
    # New y basis = [0.0, 1.0, 0.0]

    # Doubled params = 4
    # Covariance = [[nan, nan], [nan, nan]]
    # Log-likelihood = None
    # BIC = None

    return GeoCluster(data, indices, center)


@pytest.fixture
def dummy_gaussian_data() -> np.ndarray:
    # Specifically generated Gaussian (looks worse with only 20 points)
    # Max radius of ~8, centered on (0, 0), roughly back side of Earth
    return np.array(
        [
            [2.21129544, 4.06516015, 6371],
            [-1.20674765, 2.5639804, 6371],
            [7.9739952, -0.35545157, 6371],
            [1.13161905, 5.83633289, 6371],
            [-2.34134797, 3.52816964, 6371],
            [-3.86947901, 2.19552065, 6371],
            [1.43578209, -2.36003473, 6371],
            [4.55810278, -1.63904138, 6371],
            [2.57601342, 0.633518, 6371],
            [2.48063007, -0.42697437, 6371],
            [-1.38786099, 2.995909, 6371],
            [-5.46392972, 0.60650364, 6371],
            [-4.06481673, 3.68976343, 6371],
            [-8.50376463, -3.12703698, 6371],
            [1.77022762, 2.45091693, 6371],
            [0.48385653, 1.83432355, 6371],
            [1.36121698, -4.64934222, 6371],
            [7.31849999, -3.57052352, 6371],
            [6.9237546, -2.36465891, 6371],
            [-0.09451359, 4.37929612, 6371],
        ]
    )


def test_geo_cluster(
    dummy_cluster: GeoCluster,
    dummy_distant_normal_cluster: GeoCluster,
    dummy_vertical_cluster: GeoCluster,
) -> None:
    # Check indicies passed through
    np.testing.assert_array_equal(dummy_cluster.indices, np.array([0, 1, 2, 3, 4]))

    # Check df hardcoding
    assert dummy_cluster.df == 4

    # Check that a regular cluster works fine
    np.testing.assert_allclose(
        dummy_cluster.cov, np.array([[6.4, -1.097], [-1.097, 3.133]]), atol=ABS_TOL
    )
    assert dummy_cluster.log_likelihood == pytest.approx(-23.155, abs=ABS_TOL)
    assert dummy_cluster.bic == pytest.approx(52.748, abs=ABS_TOL)

    # Check that a longer plane normal changes nothing
    # Notice that the centroid loses its "height", so the BIC is unchanged
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
    np.testing.assert_allclose(
        dummy_vertical_cluster.cov, np.array([[9.5, -1.5], [-1.5, 0.3]]), atol=ABS_TOL
    )
    assert dummy_vertical_cluster.log_likelihood == pytest.approx(-73.246, abs=ABS_TOL)
    assert dummy_vertical_cluster.bic == pytest.approx(152.929, abs=ABS_TOL)


def test_bad_geo_clusters(
    dummy_thin_cluster: GeoCluster,
    dummy_1d_cluster: GeoCluster,
    dummy_0d_cluster: GeoCluster,
    dummy_non_invertible_cluster: GeoCluster,
    dummy_nan_cluster: GeoCluster,
) -> None:
    # This should succeed, but their log-likelihood and BIC will fail

    # Check cluster with thin-ish distribution (a very low cov eigenvalue)
    np.testing.assert_allclose(
        dummy_thin_cluster.cov, np.array([[9.5, 0.0], [0.0, 0.0]]), atol=ABS_TOL
    )
    assert dummy_thin_cluster.log_likelihood is None
    assert dummy_thin_cluster.bic is None

    # Check cluster with 1 dimension of distribution
    np.testing.assert_allclose(
        dummy_1d_cluster.cov, np.array([[9.5, 0.0], [0.0, 0.0]]), atol=ABS_TOL
    )
    assert dummy_1d_cluster.log_likelihood is None
    assert dummy_1d_cluster.bic is None

    # Check cluster with 0 dimensions of distribution (stacked points)
    np.testing.assert_allclose(
        dummy_0d_cluster.cov, np.array([[0.0, 0.0], [0.0, 0.0]]), atol=ABS_TOL
    )
    assert dummy_0d_cluster.log_likelihood is None
    assert dummy_0d_cluster.bic is None

    # Check that it survives a LinAlgError, using a known triggering cluster
    np.testing.assert_allclose(
        dummy_non_invertible_cluster.cov,
        np.array([[2.0, 2.0], [2.0, 2.0]]),
        atol=ABS_TOL,
    )
    assert dummy_non_invertible_cluster.log_likelihood is None
    assert dummy_non_invertible_cluster.bic is None

    # Check that it survives a NaN covariance matrix
    np.testing.assert_array_equal(
        dummy_nan_cluster.cov, np.array([[np.nan, np.nan], [np.nan, np.nan]])
    )
    assert dummy_nan_cluster.log_likelihood is None
    assert dummy_nan_cluster.bic is None


def test_geo_x_means(dummy_gaussian_data: np.ndarray) -> None:
    # Arrange 4 clusters, set at Earth distance
    # 3 and 4 are intentionally close-ish
    raw_data_1 = dummy_gaussian_data + np.array([0, 50, 0])  #  North
    raw_data_2 = dummy_gaussian_data + np.array([50, 0, 0])  #  East
    raw_data_3 = dummy_gaussian_data  # Center
    raw_data_4 = dummy_gaussian_data + np.array([-20, -20, 0])  # SW
    full_dataset = np.concat([raw_data_1, raw_data_2, raw_data_3, raw_data_4])

    # Assume k-means part works, so only check x-means logic and consistency
    # Test that center and south-west are be lumped together at k=3
    x_means = GeoXMeans(
        k_max=3, k_init=3, init="k-means++", random_state=0, n_init=10
    ).fit(full_dataset)
    assert len(np.unique(x_means.labels_)) == 3
    np.testing.assert_array_equal(
        x_means.labels_, np.array([2] * 20 + [1] * 20 + [0] * 40)
    )

    # Test it can grow from default 2 to capped k=4
    x_means = GeoXMeans(k_max=4, init="k-means++", random_state=0, n_init=10).fit(
        full_dataset
    )
    assert len(np.unique(x_means.labels_)) == 4
    np.testing.assert_array_equal(
        x_means.labels_, np.array([3] * 20 + [2] * 20 + [0] * 20 + [1] * 20)
    )

    # NOTE: Once again, labels are arbitrary, but failures due to swapped
    # labels are a sign that something changed...

    # Test that it doesn't overgrow
    x_means = GeoXMeans(init="k-means++", random_state=0, n_init=10).fit(full_dataset)
    assert len(np.unique(x_means.labels_)) == 12  # Equality for regression?


# NOTE: BIC can definitely seem better on semi-thin and small clusters, but
# that is often because the points are closer and the centroid is great. You
# can definitely see it get worse with a bad centroids or data distribution.

# TODO: We know single data points are handled, so can add that check later
# We also know that 0,0,0 normal is handled fine at least for now
# important regression test, 
