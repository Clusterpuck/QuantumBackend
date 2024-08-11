# NOTE: Might unify under a VRPSolver interface in the future for similar heuristics

import numpy as np
from sklearn.cluster import KMeans

from pydantic_models import Order
from route_optimisation.distance_matrix.distance_finder import DistanceFinder
from route_optimisation.route_solver.route_solver import RouteSolver


class RecursiveCFRS:

    def __init__(self, dm: DistanceFinder, rs: RouteSolver, split_threshold: int):
        """
        Inits an VRP solver that uses a cluster-first route-second heuristic.
        The routing half allows recursive subclustering for oversize clusters.

        TODO: May allow different clustering options for the initial vehicle
        partitioning in the future (eg. xmeans partition, kmeans subdivision).

        Parameters
        ----------
        dm : DistanceFinder
            Strategy for generating asymmetric distance matrices.
        rs : RouteSolver
            Strategy for solving ATSPs.
        split_threshold : int
            Max cluster size allowed to solve on. Trade-off between accuracy,
            solving time and solver-specific limitations.
        """
        self.__dm = dm
        self.__rs = rs
        self.__split_threshold = split_threshold

    # Helper functions
    def __cartesian_cluster(self, orders: list[Order], k: int) -> np.ndarray:
        """
        Clusters orders by their Cartesian distances, via K-means++.

        NOTE: Might be good to dedicate a strategy pattern to this in the
        future.

        Parameters
        ----------
        orders: list of Order
            Contains all x, y, z points.

        Returns
        -------
        ndarray
            1D, input-aligned array labelled by their cluster
            (eg. [0, 0, 1, 2, 0]).
        """
        points = [[o.x, o.y, o.z] for o in orders]
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0, n_init=10).fit(
            points
        )
        return kmeans.labels_

    def __flatten_list(self, deep_list):
        # Generator to flatten arbitrarily nested lists
        for item in deep_list:
            if isinstance(item, list):
                yield from self.__flatten_list(item)
            else:
                yield item

    def __solve_route(
        self,
        orders: list[Order],
    ) -> tuple[list[Order | list], Order, Order]:
        """
        Solve for optimal node visit order, recursively splitting oversized
        clusters.

        Parameters
        ----------
        orders : list of Orders
            List of route nodes to solve.

        Returns
        -------
        solution_tree : deep list of minimal routes
            The in-progress clustering tree of subroutes, in solved order.
        route_start : Order
            The first node of the deep cluster tree.
        route_end : Order
            The last node of the deep cluster tree.
        """
        # Base case: Safe size to solve
        if len(orders) <= self.__split_threshold:
            # Solve symmetric TSP (single point nodes)
            nodes = [(o, o) for o in orders]
            distance_matrix = self.__dm.build_matrix(nodes)
            solved_labels, _ = self.__rs.solve(distance_matrix)
            # TODO: Drop cost for now, might use later if strategised

            # Reorder order objects by solution
            base_solution: list[Order] = [orders[new_i] for new_i in solved_labels]

            # Extract start/end orders for the parent's TSP convenience
            route_start = base_solution[0]
            route_end = base_solution[-1]

            return base_solution, route_start, route_end

        # Recusion case: Split oversized cluster, recurse, solve cluster TSP
        spoofed_nodes = []
        unsolved_branches = []

        # Cluster and wait for subcluster TSP solutions
        cluster_labels = self.__cartesian_cluster(orders, self.__split_threshold)
        for cluster in np.unique(cluster_labels):
            order_indices = np.where(cluster_labels == cluster)[0]
            subset_orders = [orders[i] for i in order_indices]

            # Recursely gather solution subtrees and next TSP data
            subtree, subroute_start, subroute_end = self.__solve_route(subset_orders)
            unsolved_branches.append(subtree)
            spoofed_nodes.append((subroute_start, subroute_end))

        # Solve current cluster's TSP using the asymmetric node data...
        distance_matrix = self.__dm.build_matrix(spoofed_nodes)
        solved_labels, _ = self.__rs.solve(distance_matrix)
        # NOTE: Drop cost for now, might use later if also building costs

        # ...And encode this into the solution tree via branch ordering
        solution_tree = [unsolved_branches[new_i] for new_i in solved_labels]

        # Finally, spoof next TSP node by attaching route start/end data
        # Aka start coords of the spoofed route's start, and opposite case for end
        route_start = spoofed_nodes[solved_labels.index(0)][0]
        route_end = spoofed_nodes[solved_labels.index(len(solved_labels) - 1)][1]

        return solution_tree, route_start, route_end

    # Public methods
    def solve_vrp(
        self,
        orders: list[Order],
        vehicle_count: int,
    ) -> tuple[list[list[Order]], list[list[Order | list]]]:
        """
        Partition into vehicle routes, then solve for heuristically optimal
        visit order via recursive clustering.

        Parameters
        ----------
        orders : list of Orders
            List of nodes to partition up and solve on.
        vehicle_count : int
            Number of vehicles to generate routes for.

        Returns
        -------
        vehicle_routes : list of routes
            Unordered vehicle routes, each of which are ordered Orders.
        raw_tree : deep list of minimal routes
            The clustering tree of routes, in solved order (ignoring vehicles).
        """
        # Cluster into distinct vehicle routes
        vehicle_routes = []
        cluster_labels = self.__cartesian_cluster(orders, vehicle_count)

        # Gather each solution subtrees and TSP data (recurse as much as needed)
        # Start up recursion for each vehicle
        vehicle_routes = []
        raw_tree = []
        for cluster in np.unique(cluster_labels):
            order_indices = np.where(cluster_labels == cluster)[0]
            subset_orders = [orders[i] for i in order_indices]

            # Start up recursion for each vehicle
            route_tree, _, _ = self.__solve_route(subset_orders)
            raw_tree.append(route_tree)

            # If subdivided from recursion, flatten into final routes
            vehicle_routes.append([o for o in self.__flatten_list(route_tree)])

        # But will also return raws for display/debug
        return vehicle_routes, raw_tree
