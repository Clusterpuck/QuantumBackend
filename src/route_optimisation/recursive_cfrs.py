"""Vehicle Routing Problem (VRP) Solver using cluster-first route-second heuristic (CFRS)"""

# NOTE: Might unify under a VRPSolver interface in the future for similar heuristics
from typing import Generator
import numpy as np

from pydantic_models import Order
from route_optimisation.clusterer.clusterer import Clusterer
from route_optimisation.distance_matrix.distance_finder import DistanceFinder
from route_optimisation.route_solver.route_solver import RouteSolver


class RecursiveCFRS:
    """Solves VRP via CFRS using recursive clustering"""

    def __init__(
        self,
        vehicle_clusterer: Clusterer,
        subclusterer: Clusterer,
        distance_finder: DistanceFinder,
        route_solver: RouteSolver,
        max_solve_size: int,
    ):
        """
        Inits an VRP solver that uses a cluster-first route-second heuristic.
        The routing half allows recursive subclustering for oversize clusters.

        Parameters
        ----------
        vehicle_clusterer : Clusterer
            Strategy for finding distinct vehicle routes.
        subclusterer : Clusterer
            Strategy for recursive subdivision of routes for easier solving.
            Note that this MUST have a max cap, and be configured to enforce
            max_solve_size. A good example is k-means with k = max_solve_size.
        distance_finder : DistanceFinder
            Strategy for generating asymmetric distance matrices.
        route_solver : RouteSolver
            Strategy for solving ATSPs.
        max_solve_size : int
            Max cluster size allowed to solve on. Trade-off between accuracy,
            solving time and solver-specific limitations.
        """
        self.__vehicle_clusterer = vehicle_clusterer
        self.__subclusterer = subclusterer
        self.__distance_finder = distance_finder
        self.__route_solver = route_solver
        self.__max_solve_size = max_solve_size

    # Helper functions
    def __flatten_list(self, deep_list: list) -> Generator[Order, None, None]:
        """
        Generator to flatten arbitrarily nested lists

        Parameters
        ----------
        deep_list : list
            list containing nested lists

        Yields
        ------
        item: Order
            Order from nested list
        """
        for item in deep_list:
            if isinstance(item, list):
                yield from self.__flatten_list(item)
            else:
                yield item

    def __solve_route(
        self, orders: list[Order]
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
        # Base case 1: Safe size to solve
        if len(orders) <= self.__max_solve_size:
            if len(orders) == 1:  # Trivial case
                # Also saves some API calls for some df/rs
                base_solution = orders
                route_start = route_end = orders[0]
            else:
                # Solve symmetric TSP (single point nodes)
                nodes = [(o, o) for o in orders]
                distance_matrix = self.__distance_finder.build_matrix(nodes)
                solved_labels, _ = self.__route_solver.solve(distance_matrix)
                # TODO: Drop cost for now, might use later if strategised

                # Reorder order objects by solution
                base_solution: list[Order] = [orders[new_i] for new_i in solved_labels]

                # Extract start/end orders for the parent's TSP convenience
                route_start = base_solution[0]
                route_end = base_solution[-1]

            return base_solution, route_start, route_end

        # Continue: Attempt to subdivide via clustering
        cluster_labels = self.__subclusterer.cluster(orders)
        # NOTE: We just have to trust that the subclusterer enforces a max
        # split cap to equal max_solve_size. Can't validate in here atm.

        # Base case 2: Unsplittable (likely insignificant distance)
        if len(np.unique(cluster_labels)) == 1:
            # If no change and still oversized, it's likely the data are all
            # duplicates or too similar. Though not always true, there's no
            # good choice but to just return it as if "trivially solved".
            return orders, orders[0], orders[-1]

        # Recusion case: Recurse subclusters, then stitch this cluster via ATSP
        spoofed_nodes = []
        unsolved_branches = []

        # Wait for subcluster TSP solutions
        for cluster in np.unique(cluster_labels):
            order_indices = np.where(cluster_labels == cluster)[0]
            subset_orders = [orders[i] for i in order_indices]

            # Recursely gather solution subtrees and next TSP data
            subtree, subroute_start, subroute_end = self.__solve_route(subset_orders)
            unsolved_branches.append(subtree)
            spoofed_nodes.append((subroute_start, subroute_end))

        # Solve current cluster's TSP using the asymmetric node data...
        distance_matrix = self.__distance_finder.build_matrix(spoofed_nodes)
        solved_labels, _ = self.__route_solver.solve(distance_matrix)
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
        self, orders: list[Order]
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
        cluster_labels = self.__vehicle_clusterer.cluster(orders)

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
