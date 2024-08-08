from distance_matrix.distance_matrix import DistanceMatrix
from route_solver.route_solver import RouteSolver
import numpy as np

class TreeNode:
    # TODO Remove customers
    def __init__(self, cluster_id, customers=None, route=None):
        self.id = cluster_id
        self.customers = []
        self.route = route
        self.children = []
        self.cost = None

    # Getters
    def get_id(self):
        return self.id

    def get_customers(self):
        return self.customers
    
    def get_route(self):
        return self.route
    
    def get_children(self):
        return self.children
    
    def get_cost(self):
        return self.cost

    # Setters (Only using add_child and add_customer currently, 17/07/2024)
    def add_child(self, child_node):
        if not isinstance(child_node, TreeNode):
            raise TypeError("child_node must be a TreeNode")
        self.children.append(child_node)

    def add_customer(self, customer):
        self.customers.append(customer)

    def set_customers(self, customers):
        self.customers = customers

    def post_order_dfs2(self, distance_matrix, route_solver, runsheet_dictionary): #Feed in a DistanceMatrix, RouteSolver
        self._post_order_dfs_helper2(self, distance_matrix, route_solver, runsheet_dictionary)

    def _post_order_dfs_helper2(self, node, distance_matrix, route_solver, runsheet_dictionary):
        for child in node.children:
            self._post_order_dfs_helper2(child, distance_matrix, route_solver, runsheet_dictionary)
        if node.get_id() != "root":
            node.solve_node(distance_matrix, route_solver, runsheet_dictionary)

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.id) + ": " + repr(self.customers) + repr(self.route) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

    def solve_node(self, distance_matrix: DistanceMatrix, route_solver: RouteSolver, runsheet_dictionary):
        if self.children == []:
            x = distance_matrix.build_leaf_matrix(self, runsheet_dictionary)
            y = route_solver.solve(x)
            customers = self.get_customers()
            optimal_route = []
            #TODO enumerating without using data
            for index, item in enumerate(y[0]):
                optimal_route.append(customers[index])
            self.route = optimal_route
            self.cost = y[1]

        else:
            #TODO Here is the customer/route redundancy
            x = distance_matrix.build_parent_matrix(self, runsheet_dictionary)
            y = route_solver.solve(x)
            customers = self.get_customers()
            children = self.get_children()
            Alist = np.empty(len(children), dtype=object)
            for idx, data in enumerate(y):
                Alist[idx] = children[idx].get_route()
            Blist = []
            for i in Alist:
                for j in i:
                    Blist.append(j)

            #NOTE: IDK what's happening from here downwards
            optimal_route = Blist
            self.customers = Blist
            self.route = Blist 
            self.cost = y[1]

