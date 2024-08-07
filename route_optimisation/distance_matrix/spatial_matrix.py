from distance_matrix.distance_matrix import DistanceMatrix
from tree_node import TreeNode
from geographic_processing import geographic_array, geographic_to_cartesian

import numpy as np
import pandas as pd
from math import sqrt

class SpatialMatrix(DistanceMatrix):

    #TODO Fix this mess
    def build_parent_matrix(self, node: TreeNode, runsheet_dictionary):
        nodes = node.get_children()
        start_points = []
        end_points = []
        internal_costs = []
        for x in nodes:
            start_points.append(x.get_customers()[0])
            end_points.append(x.get_customers()[-1])
            internal_costs.append(x.get_cost())

        n = len(start_points)
        matrix = np.zeros((n, n), dtype=float)

        mydataset = {
            'ID': [],
            'Latitude': [],
            'Longitude': []
        }

        for x in start_points:
            key = x
            value = runsheet_dictionary.get(key)
            mydataset['ID'].append(key)
            mydataset['Latitude'].append(value[0])
            mydataset['Longitude'].append(value[1])
        df1 = pd.DataFrame(mydataset)

        mydataset = {
            'ID': [],
            'Latitude': [],
            'Longitude': []
        }

        for x in end_points:
            key = x
            value = runsheet_dictionary.get(key)
            mydataset['ID'].append(key)
            mydataset['Latitude'].append(value[0])
            mydataset['Longitude'].append(value[1])
        df2 = pd.DataFrame(mydataset)

        geo_array1 = geographic_array(df1)        # np.array: [[Latitude,Longitude]]
        cartesian_array1 = geographic_to_cartesian(geo_array1)
        geo_array2 = geographic_array(df2)        # np.array: [[Latitude,Longitude]]
        cartesian_array2 = geographic_to_cartesian(geo_array2)
        
        # loop over coords to build matrix
        for i in range(len(end_points)):
            for j in range(len(start_points)):
                if i == j:
                    matrix[i][j] = internal_costs[i]
                else:
                    matrix[i][j] = self.__get_3d_distance(cartesian_array2[i], cartesian_array1[j])
        return matrix

    #TODO: Fix this mess, looks terrible
    def build_leaf_matrix(self, node: TreeNode, runsheet_dictionary):
        # Implement matrix creation
        customers = node.get_customers()
        n = len(customers)
        matrix = np.zeros((n, n), dtype=float) # Create n x n zero-filled array

        mydataset = {
            'ID': [],
            'Latitude': [],
            'Longitude': []
        }
        
        for x in node.get_customers():
            key = x
            value = runsheet_dictionary.get(key)
            mydataset['ID'].append(key)
            mydataset['Latitude'].append(value[0])
            mydataset['Longitude'].append(value[1])
        subsheet = pd.DataFrame(mydataset)
        
        df = pd.DataFrame(customers)
        geo_array = geographic_array(subsheet)        # np.array: [[Latitude,Longitude]]
        cartesian_array = geographic_to_cartesian(geo_array)      # np.array: [[x,y,z]]

        # loop over coords to build matrix
        for i in range(len(customers)):
            for j in range(len(customers)):
                if i == j:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = self.__get_3d_distance(cartesian_array[i], cartesian_array[j])
        return matrix

    #TODO: Can fully document, simple function
    def __get_3d_distance(self, p1 : np.ndarray, p2: np.ndarray):

        # Differences between 2 points on each axes
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        delta_z = p2[2] - p1[2]
        
        distance = sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        return distance
