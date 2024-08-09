import numpy as np
from geographic_processing import delivery_list_to_cartesian, get_cartesian
from customer_allocation.customer_assignment import k_means, k_means2
from tree_node import TreeNode
import pandas as pd
from collections import OrderedDict


def partition_routes(allocation_array, split_threshold, delivery_dictionary, 
                     distance_matrix, route_solver):
    """
    Partition route into sub routes so they can be solved.

    Parameters
    ----------
    allocation_array: numpy.ndarray
        List of point allocations
    split_threshold: int
        How many clusters should exist for route partitioning
    delivery_dictionary: pandas.DataFrame
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
    new_array = partition(copy, delivery_dictionary, 0, new_array, split_threshold, None, tree, distance_matrix, route_solver) #TODO Problem here
    #new_array = partitionV2(allocation_array, delivery_dictionary, split_threshold)
    print("new_array", new_array)
    return tree

def partition(allocation_array, delivery_list, cluster_number, new_array, split_threshold, 
              new_clusters=None, cluster_tree: TreeNode=None, dm=None, rs=None):
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
            temp_array = cluster_node(allocation_array, delivery_list, cluster, split_threshold) #TODO What does this do
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

"""def partitionV2(allocation_array, delivery_dictionary, split_threshold, processed_points=None):
    copy = allocation_array.copy()
    print("Copy:", copy)

    if processed_points is None:
        processed_points = []
    
    new_allocation = allocation_array.copy()

    for cluster in np.unique(copy):
        #print(cluster)
        if cluster not in processed_points:
            points = np.where(copy == cluster)[0]
            #print(points)
            # I need to split
            if points.size > 2:
                new_allocation = cluster_nodeV2(copy, delivery_dictionary, cluster, split_threshold)
                thing = find_added_values(copy, new_allocation)
                print("thing", thing)
                new_allocation = partitionV2(new_allocation, delivery_dictionary, split_threshold, processed_points)
        else:
            processed_points.append(cluster)
    print(processed_points)
    return new_allocation"""
            



# TODO: This is what needs to be changed
def cluster_node(allocation_array, delivery_dictionary, cluster, split_threshold):
    """
    Cluster a node into sub nodes

    Parameters
    ----------
    allocation_array: numpy.ndarray
        List of point allocations
    delivery_dictionary: OrderedDict
        key = customerID
        value = (latitude, longitude)
    cluster: int
        the cluster id
    split_threshold: int
        k value for split

    Returns
    -------
    temp_array: numpy.ndarray
        new allocation of nodes after split
    """
    sub_list = create_sub_list(allocation_array, delivery_dictionary, cluster)
    cartestian_array = delivery_list_to_cartesian(sub_list)

    # If greater than one point
    if cartestian_array.shape[0] > 1:
        output = k_means(split_threshold, cartestian_array)
    else:
        output = [0]

    # for each entry in the allocation array, if it matches our current cluster, change the value to the output value
    #temparray = allocation_array.copy()
    cluster_value = max(allocation_array) + 1
    temparray = np.copy(allocation_array)
    output_tracker = 0
    for counter, x in enumerate(allocation_array):
        if x == cluster:
            temparray[counter] = output[output_tracker] + cluster_value
            output_tracker += 1
    return temparray

"""def cluster_nodeV2(allocation_array, delivery_dictionary: OrderedDict, cluster, split_threshold):
    # Your job is to do k-means again
    # I need to check the allocation_array to get points that belong to the cluster
    indices = np.where(allocation_array == cluster)[0]
    print("=======S")
    print(allocation_array)
    print(cluster)
    
    print("=======E")

    #for x in allocation_array:
    #    for indices in [index for index, (key, value) in enumerate(delivery_dictionary.items())]:
    #        print(x, indices)
    #indices = [index for index, (key, value) in enumerate(delivery_dictionary.items())]
    Alist = []
    for index in [index for index, (key, value) in enumerate(delivery_dictionary.items())]:
        if allocation_array[index] == cluster:
            #print(index, cluster)
            items_list = list(delivery_dictionary.items())
            key, value = items_list[index]
            Alist.append((key,value))
            #print(key, value)
    #NOTE We can introduce a sort here if we wish for consistency. Should work without
    #print(Alist)
    # We now have a list of points to be split
    # need a list of cartesian coordinates
    # First need to iterate over 
    #cartesian_array = np.full(len(Alist), fill_value = -1.0, dtype = float)
    print(Alist)

    output_list = [(id, get_cartesian(lat, lon)) for id, (lat, lon) in Alist]
    new_allocation = k_means2(2, output_list)
    print(output_list)
    print(allocation_array)
    print(new_allocation)
    updated_allocation = np.copy(allocation_array)
    idx = 0
    output_tracker = 0
    for x in allocation_array:
        if(x == cluster):
            updated_allocation[idx] = max(allocation_array) + 1 + new_allocation[output_tracker]
            output_tracker += 1
        idx += 1
        #if
    print(updated_allocation)
    return updated_allocation"""




def create_sub_list(allocation_array, delivery_dictionary, cluster_number):
    """
    Recursively partition routes until every node is solved

    Parameters
    ----------
    allocation_array: numpy.ndarray
        List of point allocations
    delivery_dictionary: OrderedDict
        key = customerID
        value = (latitude, longitude)
    cluster_number: int
        the cluster id

    Returns
    -------
    sub_list: pandas.DataFrame
        dataframe containing entries of the current
    """
    dataset = {
        'ID': [],
        'Latitude': [],
        'Longitude': []
    }
    
    for x in np.where(allocation_array == cluster_number)[0]:
        key = list(delivery_dictionary.keys())[x]
        value = delivery_dictionary.get(key)
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
