"""Tree Node to structure data for route splitting"""

from distance_matrix.distance_matrix import DistanceMatrix
from route_solver.route_solver import RouteSolver

class TreeNode:
    """
    Enables the construction of a generic tree to store route splitting

    Parameters
    ----------
    cluster_id : int
        The id of the node

    Attributes
    ----------
    cluster_id : int
        The id of the node
    customers : list
        Contains the list of customers for the node
    children : type, optional
        Contains the list of children TreeNodes for this node
    cost : type, optional
        Contains the cost of the route for the node

    Methods
    -------
    get_id()
        Get ID of node
    get_customers()
        Get customer list
    get_children()
        Get children nodes
    get_cost()
        Get cost of node
    add_child(child_node)
        add a child node to this node
    add_customer(customer)
        add a customer to customer list
    solve_node(distance_matrix, route_solver, delivery_dictionary)
        Find the optimal route and cost for a node
    """
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.customers = []
        self.children = []
        self.cost = None

    # Getters
    def get_id(self):
        """
        Get ID of node

        Returns
        -------
        int
            the id of the tree node            
        """
        return self.cluster_id

    def get_customers(self):
        """
        Get customer list

        Returns
        -------
        list
            the list of customers for this node            
        """
        return self.customers

    def get_children(self):
        """
        Get children nodes

        Returns
        -------
        list of TreeNodes
            the list of children TreeNodes           
        """
        return self.children

    def get_cost(self):
        """
        Get cost of node

        Returns
        -------
        float
            the cost of the route          
        """
        return self.cost

    # Setters (Only using add_child and add_customer currently, 17/07/2024)
    def add_child(self, child_node):
        """
        Add a child node to this node

        Parameters
        ----------
        child_node: TreeNode
            child node to be added

        Raises
        ------
        TypeError
            If input parameter is not a TreeNode         
        """
        if not isinstance(child_node, TreeNode):
            raise TypeError("child_node must be a TreeNode")
        self.children.append(child_node)

    def add_customer(self, customer):
        """
        add a customer to customer list

        Parameters
        ----------
        customer: int
            the customer ID to be added         
        """
        self.customers.append(customer)

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.cluster_id) + ": " + repr(self.customers) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

    def solve_node(self,
                   distance_matrix: DistanceMatrix,
                   route_solver: RouteSolver,
                   delivery_dictionary):
        """
        Find the optimal route and cost for a node

        Parameters
        ----------
        distance_matrix: DistanceMatrix
            Strategy to create the distance matrix
        route_solver: RouteSolver
            Strategy to finding the optimal route
        delivery_dictionary: pandas.DataFrame
            Dataframes containing Customer IDs, latitude and longitude        
        """
        # If child node
        if not self.children:
            matrix = distance_matrix.build_leaf_matrix(self, delivery_dictionary)
            route_allocation = route_solver.solve(matrix)

        # If parent node
        else:
            matrix = distance_matrix.build_parent_matrix(self, delivery_dictionary)
            route_allocation = route_solver.solve(matrix)
            children = self.get_children()

            # A list containing list of customers from every child node, unordered
            children_customers = [child.get_customers() for child in children]
            # Reorder the children_customers according to the order of the route_allocation
            ordered_route = [children_customers[i] for i in route_allocation[0]]
            # Flatten the route so it's a list of customers in order
            flattened_route = [item for sublist in ordered_route for item in sublist]

            self.customers = flattened_route
            self.cost = route_allocation[1]
