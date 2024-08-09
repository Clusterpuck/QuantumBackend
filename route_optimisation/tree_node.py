from distance_matrix.distance_matrix import DistanceMatrix
from route_solver.route_solver import RouteSolver
import numpy as np

class TreeNode:

    def __init__(self, cluster_id):
        self.id = cluster_id
        self.customers = []
        self.children = []
        self.cost = None

    # Getters
    def get_id(self):
        return self.id

    def get_customers(self):
        return self.customers
    
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

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.id) + ": " + repr(self.customers) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

    def solve_node(self, distance_matrix: DistanceMatrix, route_solver: RouteSolver, runsheet_dictionary):
        # If child
        if self.children == []:
            matrix = distance_matrix.build_leaf_matrix(self, runsheet_dictionary)
            y = route_solver.solve(matrix)

        # If parent node
        else:
            matrix = distance_matrix.build_parent_matrix(self, runsheet_dictionary)
            y = route_solver.solve(matrix)
            #TODO: BUG Here. I'm currently just combining in order rather checking y's optimal route
            print(y)

            """children = self.get_children()
            Alist = np.empty(len(children), dtype=object)

            for idx in range(len(y[0])):
                Alist[idx] = children[idx].get_customers()
            Blist = []
            for i in Alist:
                for j in i:
                    Blist.append(j)

            self.customers = Blist
            self.cost = y[1]"""

            children = self.get_children()
            # This should put the customers into the list
            Alist = [child.get_customers() for child in children]
            for child in children:
                pass
                #print(child.get_customers())
            print("End loop")
            print(Alist)
            print(y)
            print(y[0])

            Blist = []
            for sublist in Alist:
                #print(sublist)
                for customer in sublist:
                    #print(customer)
                    Blist.append(customer)
            print("END Other loop")

            # Equivlent structure
            Blist = [customer for sublist in Alist for customer in sublist]
            #######################
            
            print(Blist)
            self.customers = Blist
            self.cost = y[1]

