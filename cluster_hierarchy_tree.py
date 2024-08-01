from cluster_tree_solver import solve_tree, brute_force_solve
from distance_matrix.distance_matrix import DistanceMatrix
from route_solver.route_solver import RouteSolver

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
        node.solve_node(distance_matrix, route_solver)

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.id) + ": " + repr(self.customers) + "\n"
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
            print("y", y)
            #Solve Leaf
        else:
            print("Parent")
            #Solve Parent
        
        
        
        
        # The post-order stuff
        # on "result.append(node.id)", call solve
        # solve needs to:
            # Get distance matrix (Leaf is diff from parents)
            # Find shortest path
            # chuck in that optimal path in the node's route
            # chuck in that corresponding cost
