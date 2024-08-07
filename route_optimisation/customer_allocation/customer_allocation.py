import numpy as np
import pandas as pd
import pyodbc
import os
from sklearn.cluster import KMeans
from collections import OrderedDict


from customer_allocation.validation import validate_inputs
from geographic_processing import geographic_array, geographic_to_cartesian
    
def get_customer_allocation(runsheet, k):
    valid = False
    try:
        connection_string = os.getenv('QuantumTestString')
        validate_inputs(runsheet,k,connection_string)
        valid = True
    except (TypeError, ValueError, pyodbc.DatabaseError) as ex:
        print(ex)
    if valid:
        cartesian_array = runsheet_to_cartesian(runsheet, connection_string)
        return get_customer_allocation2(k, cartesian_array)
    else:
        print("Input was not valid")

# This should be moved elsewhere
def runsheet_to_cartesian(runsheet, connection_string):
    geo_array = geographic_array(runsheet,connection_string)        # np.array: [[Latitude,Longitude]]
    cartesian_array = geographic_to_cartesian(geo_array)
    return cartesian_array 

# Create a dictionary that will map indices to ID
# Store it in memory
def create_dictionary(dataframe):
    #return {row[0]: row[1] for row in dataframe.values} #OLD CODE WORKS, UNORDERED DICTIONARY
    return OrderedDict((row[0], row[1]) for row in dataframe.values)

# Do K-means++ on geo_array
# Output: [Index,assignment]
#TODO rename it to something generic
def get_customer_allocation2(k, geo_array):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(geo_array)

    labels = kmeans.labels_
    return labels