"""Module Text"""

from collections import OrderedDict
from sklearn.cluster import KMeans

from geographic_processing import runsheet_to_cartesian
from .validation import validate_inputs

# TODO: Fix the return, might return None
def get_customer_allocation(delivery_list, k):
    valid = False
    try:
        valid = validate_inputs(delivery_list,k)
    except (TypeError, ValueError) as ex:
        print(ex)
    if valid:
        cartesian_array = runsheet_to_cartesian(delivery_list)
        return k_means(k, cartesian_array)
    else:
        print("Input was not valid")

def create_dictionary(df):
    return OrderedDict((int(row[0]), (row[1], row[2])) for row in df.values)

# Do K-means++ on geo_array
# Output: [assignment]
def k_means(k, geo_array):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(geo_array)

    labels = kmeans.labels_
    return labels
