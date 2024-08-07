from sklearn.cluster import KMeans
from collections import OrderedDict

from customer_allocation.validation import validate_inputs # TODO: Important for later
from geographic_processing import runsheet_to_cartesianV2

# runsheet = ID, Lat, Long
# TODO: Fix the return, might return None
def get_customer_allocationV2(runsheet, k):
    valid = False
    try:
        #validate_inputs(runsheet,k,connection_string) #TODO: Important for later
        valid = True
    except (TypeError, ValueError) as ex:
        print(ex)
    if valid:
        cartesian_array = runsheet_to_cartesianV2(runsheet)
        return k_means(k, cartesian_array)
    else:
        print("Input was not valid")

def create_dictionaryV2(df):
    return OrderedDict((int(row[0]), (row[1], row[2])) for row in df.values)

# Do K-means++ on geo_array
# Output: [assignment]
def k_means(k, geo_array):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(geo_array)

    labels = kmeans.labels_
    return labels