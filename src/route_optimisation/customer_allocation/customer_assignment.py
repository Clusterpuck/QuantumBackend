"""Assign customers to a route based on geographic distance via K-means++"""
# NOTE: Consider renaming customer to order for semantics

import numpy as np
from sklearn.cluster import KMeans

from geographic_processing import orders_to_cartesian
from pydantic_models import Order
from .validation import validate_inputs

def cartesian_cluster(orders: list[Order], k: int) -> np.ndarray:
    """
    Validate the parameters and cluster points based on geographic location

    Parameters
    ----------
    orders: list of Order
        Dataframes containing Customer IDs, latitude and longitude
    k: int
        How many routes must be created

    Returns
    -------
    customer_assignment: numpy.ndarray
        List of point allocations
        [0, 1, 0] means cluster 0 has point 0 and point 2, cluster 1 has point 1
        Index of allocation and delivery_list represent the same customer
    """
    # Perform cartesian clustering
    cartesian_array = orders_to_cartesian(orders)
    customer_assignment = k_means(k, cartesian_array)
    return customer_assignment

def k_means(k, cartesian_array: np.ndarray) -> np.ndarray:
    """
    Uses k-means++ on an array of cartesian coordinates and clusters them into k clusters.

    Parameters
    ----------
    k: int
        How many clusters should there be
    cartesian_array : numpy.ndarray of numpy.ndarray of floats
        An array of arrays of floats representing 3D positions
        [[x1,y1,z1], [x2,y2,z2]] for 2 points

    Returns
    -------
    labels: numpy.ndarray
        List of point allocations
        [0, 1, 0] means cluster 0 has point 0 and point 2, cluster 1 has point 1
    """
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(cartesian_array)
    labels = kmeans.labels_
    return labels

# def k_means2(k, cartesian_array):
#     coordinates_only = [coords for _, coords in cartesian_array]
#     print("==============Kmeans Cartesian\n", coordinates_only)
#     print("\n==============Kmeans CartesianE")
#     kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(coordinates_only)
#     labels = kmeans.labels_
#     return labels
