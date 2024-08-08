"""Assign customers to a route based on geographic distance via K-means++"""

from collections import OrderedDict
from sklearn.cluster import KMeans

from geographic_processing import runsheet_to_cartesian
from .validation import validate_inputs

def get_customer_allocation(delivery_list, k):
    try:
        validate_inputs(delivery_list,k)
        cartesian_array = runsheet_to_cartesian(delivery_list)
        customer_assignment = k_means(k, cartesian_array)
    except IOError as ex:
        print(f"An error occured with input validation.\n{ex}")
    return customer_assignment

def create_dictionary(df):
    return OrderedDict((int(row[0]), (row[1], row[2])) for row in df.values)

# Do K-means++ on geo_array
# Output: [assignment]
def k_means(k, geo_array):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(geo_array)
    labels = kmeans.labels_
    return labels
