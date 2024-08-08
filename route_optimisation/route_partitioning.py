import numpy as np
from geographic_processing import delivery_list_to_cartesian
from customer_allocation.customer_assignment import k_means
from tree_node import TreeNode
import pandas as pd

def partition_routes(allocation_array, split_threshold, delivery_list, 
                     distance_matrix, route_solver):
    """
    Partition route into sub routes so they can be solved.

    Parameters
    ----------
    allocation_array: numpy.ndarray
        List of point allocations
    split_threshold: int
        How many clusters should exist for route partitioning
    delivery_list: pandas.DataFrame
        Dataframes containing Customer IDs, latitude and longitude
    distance_matrix: DistanceMatrixContext
        The distance matrix building method
    route_solver: RouteSolver
        The route solving method

    Returns
    -------
    tree: TreeNode
        The entire tree of routes post-partitioning
    """
    tree = TreeNode("root")
    copy = allocation_array.copy()
    new_array = np.full(allocation_array.shape[0], fill_value = -1, dtype = int)
    new_array = partition(copy, delivery_list, 0, new_array, split_threshold, None, tree, distance_matrix, route_solver) # Pass in the tree
    return tree

def partition(allocation_array, delivery_list, cluster_number, new_array, split_threshold, 
              new_clusters=None, cluster_tree=None, dm=None, rs=None):
    """
    Recursively partition routes until every node is solved

    Parameters
    ----------
    allocation_array: numpy.ndarray
        List of point allocations
    delivery_list: pandas.DataFrame
        Dataframes containing Customer IDs, latitude and longitude
    cluster_number: int
        the cluster id
    new_array: numpy.ndarray
        allocation array post-clustering
    split_threshold: int
        k value for clustering
    processed_clusters: Set
        Set of cluster ids that have been solved
    new_clusters: numpy.ndarray
        array of new clusters ids that have been added
    cluster_tree: TreeNode
        node from cluster tree
    distance_matrix: DistanceMatrixContext
        The distance matrix building method
    route_solver: RouteSolver
        The route solving method

    Returns
    -------
    new_array: numpy.ndarray
        allocation array post-clustering
    """
            
    if new_clusters is None:
        # Set unprocessed clusters to allocations, happens once
        unprocessed_clusters = allocation_array.copy()
    else:
        # Set unprocessed clusters to those that have been added from previous split
        unprocessed_clusters = new_clusters.copy()

    # for each unique cluster that has not yet been processed
    for cluster in np.unique(unprocessed_clusters):
        # Grab indices of points from cluster, store in array
        points = np.where(allocation_array == cluster)[0]
        # Cluster too big
        if points.size > split_threshold:
            comparison = allocation_array.copy()
            # Split the cluster down
            temp_array = get_subsheetV2(allocation_array, delivery_list, cluster, split_threshold) #TODO What does this do
            new_clusters = find_added_values(comparison, temp_array)

            cluster_number += 1
            
            parent = TreeNode(cluster) # Create a node for this cluster
            cluster_tree.add_child(parent) # Add to child
            # Recursively partition sub-clusters
            new_array = partition(temp_array, delivery_list, cluster_number, temp_array, split_threshold, new_clusters, parent, dm, rs)
            # Solve the node
            parent.solve_node(dm, rs, delivery_list)
        # Cluster is small enough
        else:
            # Cluster is considered processed
            #processed_clusters.add(cluster)
            if points.size > 0:
                leaf = TreeNode(cluster)
                for point in points:
                    key = list(delivery_list.keys())[point]
                    leaf.add_customer(key)
                cluster_tree.add_child(leaf)
                leaf.solve_node(dm, rs, delivery_list)
    return new_array

    


# TODO: This is what needs to be changed
def get_subsheetV2(allocation_array, runsheet_dictionary, cluster, split_threshold):

    sub_list = create_sub_list(allocation_array, runsheet_dictionary, cluster)
    cartestian_array = delivery_list_to_cartesian(sub_list)

    # If greater than one point
    if cartestian_array.shape[0] > 1:
        output = k_means(split_threshold, cartestian_array)
    else:
        output = [0]

    # for each entry in the allocation array, if it matches our current cluster, change the value to the output value
    temparray = allocation_array.copy()
    cluster_value = max(allocation_array) + 1
    output_tracker = 0
    for counter, x in enumerate(allocation_array):
        if x == cluster:
            temparray[counter] = output[output_tracker] + cluster_value
            output_tracker += 1
    return temparray

def create_sub_list(allocation_array, runsheet_dictionary, cluster):
    
    dataset = {
        'ID': [],
        'Latitude': [],
        'Longitude': []
    }
    
    for x in np.where(allocation_array == cluster)[0]:
        key = list(runsheet_dictionary.keys())[x]
        value = runsheet_dictionary.get(key)
        dataset['ID'].append(key)
        dataset['Latitude'].append(value[0])
        dataset['Longitude'].append(value[1])
    sub_list = pd.DataFrame(dataset)
    return sub_list



def find_added_values(arr1, arr2):
    """
    Find the new values in arr2 based on arr1

    Parameters
    ----------
    arr1: numpy.ndarray 
        old array
    arr2: numpy.ndarray 
        new array

    Returns
    -------
    numpy.ndarray
        Contains added values in a list
    """
    set1 = set(np.unique(arr1))
    set2 = set(np.unique(arr2))

    added = set2 - set1

    return np.array(list(added))
