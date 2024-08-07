from distance_matrix.distance_matrix import DistanceMatrix
from cluster_hierarchy_tree import TreeNode
from geographic_processing import geographic_array, geographic_to_cartesian

import numpy as np
import os
import pandas as pd
from math import sqrt

class SpatialMatrix(DistanceMatrix):

    def build_parent_matrix(self, node: TreeNode):
        # Implement matrix creation
        # query children for starts and ends
        nodes = node.get_children()
        start_points = []
        end_points = []
        internal_costs = []
        for x in nodes:
            start_points.append(x.get_route()[0])
            end_points.append(x.get_route()[-1])
            internal_costs.append(x.get_cost())

        n = len(start_points)
        matrix = np.zeros((n, n), dtype=float)

        connection_string = os.getenv('QuantumTestString')
        df1 = pd.DataFrame(start_points)
        df2 = pd.DataFrame(end_points)
        geo_array1 = geographic_array(df1,connection_string)        # np.array: [[Latitude,Longitude]]
        cartesian_array1 = geographic_to_cartesian(geo_array1)
        geo_array2 = geographic_array(df2,connection_string)        # np.array: [[Latitude,Longitude]]
        cartesian_array2 = geographic_to_cartesian(geo_array2)
        
        # loop over coords to build matrix
        for i in range(len(end_points)):
            for j in range(len(start_points)):
                if i == j:
                    matrix[i][j] = internal_costs[i]
                else:
                    matrix[i][j] = self.__get_3D_distance(cartesian_array2[i], cartesian_array1[j])
                print(matrix, end_points[i], start_points[j])
        return matrix
            
        # query for lat, long for start and end. store as [x_start, x_end, y_start, y_end, ...]? or Seperate? works either way
            # Geographic processing, geographic array
            # geographic_to_cartesian
        # loop over start coords
            # loop over end coords
                # Get cartesian distance (A + A_end -> B_start)
                # If i==j, set to 0
        # return matrix

    #TODO: Fix this mess, looks terrible
    def build_leaf_matrix(self, node):
        # Implement matrix creation
        customers = node.get_customers()
        n = len(customers)
        matrix = np.zeros((n, n), dtype=float) # Create n x n zero-filled array

        connection_string = os.getenv('QuantumTestString')
        df = pd.DataFrame(customers)
        geo_array = geographic_array(df,connection_string)        # np.array: [[Latitude,Longitude]]
        cartesian_array = geographic_to_cartesian(geo_array)      # np.array: [[x,y,z]]

        # loop over coords to build matrix
        for i in range(len(customers)):
            for j in range(len(customers)):
                if i == j:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = self.__get_3D_distance(cartesian_array[i], cartesian_array[j])
        return matrix

    #TODO: Can fully document, simple function
    def __get_3D_distance(self, p1 : np.ndarray, p2: np.ndarray):

        # Differences between 2 points on each axes
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        delta_z = p2[2] - p1[2]
        
        distance = sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        return distance
