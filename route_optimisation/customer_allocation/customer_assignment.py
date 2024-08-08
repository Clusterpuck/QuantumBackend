"""Assign customers to a route based on geographic distance via K-means++"""

from collections import OrderedDict
from sklearn.cluster import KMeans

from geographic_processing import delivery_list_to_cartesian
from .validation import validate_inputs

def get_customer_allocation(delivery_list, k):
    """
    Validate the parameters and cluster points based on geographic location

    Parameters
    ----------
    delivery_list: pandas.DataFrame
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
    try:
        validate_inputs(delivery_list,k)
        cartesian_array = delivery_list_to_cartesian(delivery_list)
        customer_assignment = k_means(k, cartesian_array)
    except IOError as ex:
        print(f"An error occured with input validation.\n{ex}")
    return customer_assignment

def create_dictionary(df):
    """
    Create a dictionary of Customer IDs and their latitude and longitude.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframes containing Customer IDs, latitude and longitude

    Returns
    -------
    OrderedDict 
        key = customerID
        value = (latitude, longitude)
    """
    return OrderedDict((int(row[0]), (row[1], row[2])) for row in df.values)

def k_means(k, cartesian_array):
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
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(cartesian_array)
    labels = kmeans.labels_
    return labels
