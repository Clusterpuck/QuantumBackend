import numpy as np
import pyodbc
import os
from sklearn.cluster import KMeans
from customer_allocation.customer_allocation import runsheet_to_cartesian, get_customer_allocation2
from cluster_hierarchy_tree import *
import pandas as pd

# Deprecated
def partition_routes(allocation_array, split_threshold, runsheet_dictionary):
    # Check validity of params
    new_array = np.full(allocation_array.shape[0], fill_value = -1, dtype = int)
    for cluster in np.unique(allocation_array):
        points = np.where(allocation_array == cluster)[0]
        #print("Points",points)
        if points.size > split_threshold:
            #print("IF")
            print("BEFORE", new_array)
            new_array = recursive_split(runsheet_dictionary, cluster, allocation_array, new_array)
            print("AFTER", new_array)
        else:
            print()
            #new_array = insert_into_array(new_array, allocation_array, cluster)

    print("FINAL new_array", new_array)
    print("PRE cluster", np.unique(allocation_array))
    print("FINAL allocation_array", allocation_array)

def partition_routes2(allocation_array, split_threshold, runsheet_dictionary):
    #DECLARE ROOT
    tree = TreeNode("root")
    print("Create Tree")
    print(allocation_array)
    copy = allocation_array.copy()
    new_array = np.full(allocation_array.shape[0], fill_value = -1, dtype = int)
    new_array = recursion3(copy, runsheet_dictionary, 0, new_array, split_threshold, None, None, tree) # Pass in the tree
    print("FINAL na", new_array)
    print("FINAL c ", copy)
    print("FINAL aa", allocation_array)
    print("Runsheet Dictionary", runsheet_dictionary)
    return tree


def recursion2(allocation_array, runsheet_dictionary, cluster_number, new_array, split_threshold):
    #if cluster_number is None:
    for cluster in np.unique(allocation_array):
        print("=================")
        points = np.where(allocation_array == cluster)[0]
        """if points.size <= split_threshold and not np.isin(cluster, allocation_array):
            print("Cluster", cluster)
            print("Points", points, "\n")
            new_array = insert_into_array(new_array, allocation_array, cluster)
        else:
            temp_array = get_subsheet(allocation_array, runsheet_dictionary, cluster)
            recursion2(allocation_array, runsheet_dictionary, None, temp_array, split_threshold)
            print("recursionTemp",temp_array)"""
        if points.size > split_threshold:
            temp_array = get_subsheet(allocation_array, runsheet_dictionary, cluster)
            print("----------------ENTER", cluster_number)
            # If you entering this, then you are trying to split. Create a parent node here, pass parent in
            new_array = recursion2(temp_array, runsheet_dictionary, cluster, temp_array, split_threshold)
            print("----------------EXIT", cluster_number)
        else:
            print(new_array)
            print("ENTERED ELSE")
            pass # Add leaf node
    return new_array

def recursion3(allocation_array, runsheet_dictionary, cluster_number, new_array, split_threshold, processed_clusters=None, other_clusters=None, cluster_tree=None):
    #if cluster_number is None:
    print("recursion3 called")
    if processed_clusters is None:
        processed_clusters = set()

    if other_clusters is None:
        thing = allocation_array.copy()
    else:
        thing = other_clusters.copy()

    #for cluster in np.unique(allocation_array):
    for cluster in np.unique(thing):

        if cluster in processed_clusters:
            continue

        points = np.where(allocation_array == cluster)[0]
        """if points.size <= split_threshold and not np.isin(cluster, allocation_array):
            print("Cluster", cluster)
            print("Points", points, "\n")
            new_array = insert_into_array(new_array, allocation_array, cluster)
        else:
            temp_array = get_subsheet(allocation_array, runsheet_dictionary, cluster)
            recursion2(allocation_array, runsheet_dictionary, None, temp_array, split_threshold)
            print("recursionTemp",temp_array)"""
        if points.size > split_threshold:
            comparison = allocation_array.copy()
            #print("xxxxxx", allocation_array)
            temp_array = get_subsheet(allocation_array, runsheet_dictionary, cluster)
            y = find_added_values(comparison, temp_array)
            print(y)
            #print("xxxxxx", temp_array)

            cluster_number += 1
            print("----------------ENTER", cluster_number, cluster)
            # If you entering this, then you are trying to split. Create a parent node here, pass parent in
            print("Create Parent node", cluster_number, cluster)
            parent = TreeNode(cluster)
            cluster_tree.add_child(parent)
            print(temp_array, cluster_number, cluster)
            new_array = recursion3(temp_array, runsheet_dictionary, cluster_number, temp_array, split_threshold, processed_clusters, y, parent)
            print("----------------EXIT", cluster_number, cluster)
        else:
            processed_clusters.add(cluster)
            if points.size > 0:
                print("Create Leaf", cluster, points)
                leaf = TreeNode(cluster)
                for point in points:
                    key = list(runsheet_dictionary.keys())[point]
                    leaf.add_customer(key)
                cluster_tree.add_child(leaf)
            pass # Add leaf node
    return new_array

def get_subsheet(allocation_array, runsheet_dictionary, cluster):
    
    #print("subcluster",cluster)
    #print("suballoc",allocation_array)

    if np.where(allocation_array == cluster)[0].size == 1:
        print()

    mydataset = {
        'ID': [],
        'Customer': []
    }

    for x in np.where(allocation_array == cluster)[0]:
        key = list(runsheet_dictionary.keys())[x]
        value = runsheet_dictionary.get(key)
        mydataset['ID'].append(key)
        mydataset['Customer'].append(value)
    subsheet = pd.DataFrame(mydataset)

    try:
        connection_string = os.getenv('QuantumTestString')
    except (pyodbc.DatabaseError) as ex:
        print(ex)
    cartestian_array = runsheet_to_cartesian(subsheet, connection_string)
    if cartestian_array.size != 3:
        output = get_customer_allocation2(2, cartestian_array)
    else:
        output = [0]

    # Do not change below to copy, does unnecessary recursion due to copy nature
    temparray = allocation_array #TODO, Screwed this up #allocation_array.copy().
    cluster_value = max(allocation_array) + 1
    output_tracker = 0
    for counter, x in enumerate(allocation_array):
        if x == cluster:
            temparray[counter] = output[output_tracker] + cluster_value
            output_tracker += 1
    #print("A", allocation_array)
    #print("t", temparray)
    return temparray



def insert_into_array(new_array, old_array, number):
    if len(new_array) != len(old_array):
        raise ValueError("Both arrays must have the same length.")
    #print(new_array)
    for i in enumerate(new_array):
        if old_array[i[0]] == number:
            new_array[i[0]] = old_array[i[0]]
    #print(new_array)
    return new_array

# If we have duplicate customers, must change the dictionary to a list or something
# ASSUMES: 1-1 relationship with ID and Customer (Both Ways)
# Could try passing the distance matrix down as param and get rid of irrelevant entries? Saves time and Db Access
# Maybe do a if none check
    # None -> get distance matrix
    # Exists -> drop irrelevant entries then do k-means
def recursive_split(runsheet_dictionary, number, allocation_array, new_array):
    #allocation_array =  np.array([0, 1, 1, 1]) #NOTE TESTING OVERRIDES
    #output =  np.array([1, 0, 1])
    print("PRE SPLIT", allocation_array)
    #print(number)
    #print(np.where(allocation_array == number)[0])
    mydataset = {
        'ID': [],
        'Customer': []
    }

    for x in np.where(allocation_array == number)[0]:
        key = list(runsheet_dictionary.keys())[x]
        value = runsheet_dictionary.get(key)
        mydataset['ID'].append(key)
        mydataset['Customer'].append(value)
    subsheet = pd.DataFrame(mydataset)
    #print("SUB",subsheet)
    #print("==================")
    #print(mydataset)
    try:
        connection_string = os.getenv('QuantumTestString')
    except (pyodbc.DatabaseError) as ex:
        print(ex)
    cartestian_array = runsheet_to_cartesian(subsheet, connection_string)
    output = get_customer_allocation2(2, cartestian_array) #TODO Rename this, it is cursed, Also k value optimisation
    #TODO GET CUSTOMERS /\
    #print(output)
    #print("==================")
    
    # Annoying to explain loop
    # Apply the split to Original array
    temparray = allocation_array
    cluster_value = max(allocation_array) + 1
    output_tracker = 0
    for counter, x in enumerate(allocation_array):
        if x == number:
            temparray[counter] = output[output_tracker] + cluster_value
            output_tracker += 1
    
    #print("------------")
    #print(temparray)
    # Do a k-means split
    # Check size of each cluster
    #   Too big, call split again on cluster
    #   Just fine, all Good

    ##########CURSED REPEATED CODE

    #new_array = np.full(allocation_array.shape[0], fill_value = -1, dtype = int)

    for cluster in np.unique(allocation_array):
        points = np.where(allocation_array == cluster)[0]
        if points.size > 1:
            print("IFR")

            new_array = recursive_split(runsheet_dictionary, cluster, allocation_array, new_array)
        else:
            #print("insert_into_arrayR", points)
            print("BEFORER", new_array)
            new_array = insert_into_array(new_array, allocation_array, cluster)
            print("AFTERR", new_array)
    ##########CURSED REPEATED CODE
    
    print()
    print()
    print()
    return temparray

def find_added_values(arr1, arr2):
    set1 = set(np.unique(arr1))
    set2 = set(np.unique(arr2))

    added = set2 - set1

    return np.array(list(added))
"""
    allocations = np.array([1, 2, 1, 1, 3])

    # Find unique clusters
    unique_clusters = np.unique(allocations)
    print(np.unique(allocations))

    # Iterate over each unique cluster
    for cluster in unique_clusters:
        # Find points in the current cluster
        points = np.where(allocations == cluster)[0]
        print(f"Cluster {cluster} has points at indices: {points}")
"""