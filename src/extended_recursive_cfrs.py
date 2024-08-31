from pydantic_models import Order
from route_optimisation.clusterer.clusterer import Clusterer
from route_optimisation.distance_matrix.distance_finder import DistanceFinder
from route_optimisation.recursive_cfrs import RecursiveCFRS
import numpy as np

from route_optimisation.route_solver.route_solver import RouteSolver

class ExtendedRecursiveCFRS(RecursiveCFRS):
    def __init__(
        self,
        vehicle_clusterer: Clusterer,
        subclusterer: Clusterer,
        distance_finder: DistanceFinder,
        route_solver: RouteSolver,
        max_solve_size: int,
    ):
        super().__init__(  # Forward parameters to the parent class
            vehicle_clusterer=vehicle_clusterer,
            subclusterer=subclusterer,
            distance_finder=distance_finder,
            route_solver=route_solver,
            max_solve_size=max_solve_size,
        )

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