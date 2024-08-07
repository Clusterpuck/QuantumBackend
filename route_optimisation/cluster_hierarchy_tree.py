from distance_matrix.distance_matrix import DistanceMatrix
from route_solver.route_solver import RouteSolver
import numpy as np

class TreeNode:
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

    def post_order_dfs(self):
        result = []
        self._post_order_dfs_helper(self, result)
        return result

    def _post_order_dfs_helper(self, node, result):
        for child in node.children:
            self._post_order_dfs_helper(child, result)
        print(node.id)
        result.append(node.id)

    def post_order_dfs2(self, distance_matrix, route_solver): #Feed in a DistanceMatrix, RouteSolver
        self._post_order_dfs_helper2(self, distance_matrix, route_solver)

    def _post_order_dfs_helper2(self, node, distance_matrix, route_solver):
        for child in node.children:
            self._post_order_dfs_helper2(child, distance_matrix, route_solver)
        if node.get_id() != "root":
            node.solve_node(distance_matrix, route_solver)

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.id) + ": " + repr(self.customers) + repr(self.route) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret
    
    def find_node_by_id(self, target_id):
        result = []
        self._find_node_by_id_helper(self, target_id, result)
        if result:
            return result[0]
        else:
            return None

    def _find_node_by_id_helper(self, node, target_id, result):
        if node.id == target_id:
            result.append(node)
        for child in node.children:
            self._find_node_by_id_helper(child, target_id, result)

    # DEPRECATED
    #def solve_tree(self):
    #    brute_force_solve(self)

    def solve_node(self, distance_matrix: DistanceMatrix, route_solver: RouteSolver):
        if self.children == []:
            print(self)
            x = distance_matrix.build_leaf_matrix(self)
            print("x", x)
            print("DONE")
            y = route_solver.solve(x)
            # Translate the index route to actual route
            customers = self.get_customers()
            optimal_route = []
            for index, item in enumerate(y[0]):
                #print("LOOPING HERE", customers[index])
                optimal_route.append(customers[index])
            self.route = optimal_route
            self.cost = y[1]
            #print("y", y)
            #Solve Leaf
        else:
            print("Parent")
            x = distance_matrix.build_parent_matrix(self)
            y = route_solver.solve(x) # Works
            print("y", y[0])
            customers = self.get_customers() #NOTE: This is the problem. Parents do not have proper routes yet. Make function to resolve.
            children = self.get_children()
            Alist = np.empty(len(children), dtype=object)
            for idx, data in enumerate(y):
                print("idx", idx)
                Alist[idx] = children[idx].get_route()
            Blist = []
            for i in Alist:
                print("i", i)
                for j in i:
                    Blist.append(j)
            print(Alist)
            print(Blist)
            print(len(Blist))

            #NOTE: IDK what's happening from here downwards
            optimal_route = Blist
            #for index, item in enumerate(y[0]):
                #print("LOOPING HERE", customers[index])
                #optimal_route.append(Blist[index])
            print("optimal_route", optimal_route)
            self.customers = Blist
            self.route = Blist #optimal_route
            self.cost = y[1]
            print("                     DID A THING")
            #Solve Parent

