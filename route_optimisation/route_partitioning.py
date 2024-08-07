import numpy as np
from geographic_processing import runsheet_to_cartesian
from customer_allocation.customer_assignment import k_means
from cluster_hierarchy_tree import *
import pandas as pd

def partition_routesV3(allocation_array, split_threshold, runsheet_dictionary, dm, rs):
    tree = TreeNode("root")
    copy = allocation_array.copy()
    new_array = np.full(allocation_array.shape[0], fill_value = -1, dtype = int)
    new_array = recursionV3(copy, runsheet_dictionary, 0, new_array, split_threshold, None, None, tree, dm, rs) # Pass in the tree
    return tree

# Recurse, build and solve in 1 loop
def recursionV3(allocation_array, runsheet_dictionary, cluster_number, new_array, split_threshold, processed_clusters=None, other_clusters=None, cluster_tree=None, dm=None, rs=None):
    if processed_clusters is None:
        processed_clusters = set()

    if other_clusters is None:
        thing = allocation_array.copy()
    else:
        thing = other_clusters.copy()

    for cluster in np.unique(thing):

        if cluster in processed_clusters:
            continue

        points = np.where(allocation_array == cluster)[0]
        if points.size > split_threshold:
            comparison = allocation_array.copy()
            temp_array = get_subsheetV2(allocation_array, runsheet_dictionary, cluster)
            y = find_added_values(comparison, temp_array)

            cluster_number += 1
            # If you entering this, then you are trying to split. Create a parent node here, pass parent in
            parent = TreeNode(cluster)
            cluster_tree.add_child(parent)
            new_array = recursionV3(temp_array, runsheet_dictionary, cluster_number, temp_array, split_threshold, processed_clusters, y, parent, dm, rs)
            parent.solve_node(dm, rs, runsheet_dictionary)
        else:
            processed_clusters.add(cluster)
            if points.size > 0:
                leaf = TreeNode(cluster)
                for point in points:
                    key = list(runsheet_dictionary.keys())[point]
                    leaf.add_customer(key)
                cluster_tree.add_child(leaf)
                leaf.solve_node(dm, rs, runsheet_dictionary)
    return new_array

    


# TODO: This is what needs to be changed
def get_subsheetV2(allocation_array, runsheet_dictionary, cluster):

    mydataset = {
        'ID': [],
        'Latitude': [],
        'Longitude': []
    }

    # Create a subsheet of customers based off of cluster assignment
    # TODO: Could be more elegant here
    for x in np.where(allocation_array == cluster)[0]:
        key = list(runsheet_dictionary.keys())[x]
        value = runsheet_dictionary.get(key)
        mydataset['ID'].append(key)
        mydataset['Latitude'].append(value[0])
        mydataset['Longitude'].append(value[1])
    subsheet = pd.DataFrame(mydataset)

    cartestian_array = runsheet_to_cartesian(subsheet)
    if cartestian_array.size != 3:
        output = k_means(2, cartestian_array)
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

def find_added_values(arr1, arr2):
    set1 = set(np.unique(arr1))
    set2 = set(np.unique(arr2))

    added = set2 - set1

    return np.array(list(added))
