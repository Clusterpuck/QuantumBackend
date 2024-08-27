"""Creates TSP QUBO for D-Wave Solver"""

import numpy as np


class TSPFormulation:
    """
    Class for constructing a Travelling Salesman Problem QUBO.

    Based on Andrew Lucas' Ising formulation and BOHRTECHNOLOGY's QUBO
    implementation. Supports asymmetric costs and paths, but not fixed
    start point (yet).
    """

    # Helper functions
    def __add_cost_objective(
        self, cost_factor: int, distance_matrix: np.ndarray, circuit: bool
    ) -> dict[tuple[int, int], int]:
        """
        Encodes asymmetric distances as QUBO edges between sequential nodes.

        Runs in O(n^3), creating n^2(n-1) new edges. If path, n(n-1)^2.

        Parameters
        ----------
        cost_factor : int
            Scaling factor to adjust distance weighting.
        distance_matrix : ndarray
            Asymmetric distance matrix to find a shortest route for.
        circuit : bool
            If True, disconnects last node from first to force path discovery.

        Returns
        -------
        dict
            Collection of QUBO edge weights, indexed by 2-tuple of BV indices.
            Note that QUBO just sums (x, y) and (y, x) edges.
        """
        qubo_dict = {}
        n = len(distance_matrix)

        # If path, do not connect end back to start. Neither reward nor
        # penalise, acting like a dummy 0 cost node without the node wastage.
        max_row = n if circuit else n - 1

        # Links each BV to the row below (represents travel cost to next node)
        for row in range(max_row):
            for col in range(n):
                qubit_a = row * n + col
                for col_iter in range(n):
                    if col != col_iter:  # Avoid one-hot violations
                        qubit_b = (row + 1) % n * n + col_iter
                        qubo_dict[(qubit_a, qubit_b)] = (
                            cost_factor * distance_matrix[col][col_iter]
                        )

        return qubo_dict

    def __add_selection_incentive(
        self, constraint_factor: int, n: int
    ) -> dict[tuple[int, int], int]:
        """
        Encourages variable selection, effectively discouraging less than n
        selections.

        Runs in O(n^2), creating n^2 new edges (may as well be self-weights).

        Parameters
        ----------
        constraint_factor : int
            Scaling factor to adjust constraint weighting.
        n : int
            Number of nodes in a route.

        Returns
        -------
        dict
            Collection of QUBO edge weights, indexed by 2-tuple of BV indices.
            Note that QUBO just sums (x, y) and (y, x) edges.
        """
        qubo_dict = {}

        # Basically, reward every individual variable selection
        for row in range(n):
            for col in range(n):
                qubit_a = row * n + col
                qubo_dict[(qubit_a, qubit_a)] = -constraint_factor

        return qubo_dict

    def __add_time_constraints(
        self, constraint_factor: int, n: int
    ) -> dict[tuple[int, int], int]:
        """
        Strongly penalises being in multiple places at the same time (aka
        horizontal one-hot violations). When paired with position constraints,
        this inherently discourages more than n selections.

        Runs in O(n^3), creating n(n-1)^2 new edges.

        Parameters
        ----------
        constraint_factor : int
            Scaling factor to adjust constraint weighting.
        n : int
            Number of nodes in a route.

        Returns
        -------
        dict
            Collection of QUBO edge weights, indexed by 2-tuple of BV indices.
            Note that QUBO just sums (x, y) and (y, x) edges.
        """
        qubo_dict = {}

        for row in range(n):
            for col in range(n - 1):  # Link cols to their right-ward BVs
                qubit_a = row * n + col
                for col_offset in range(col + 1, n):
                    qubit_b = row * n + col_offset
                    qubo_dict[(qubit_a, qubit_b)] = 2 * constraint_factor
                    # Penalty must be strong enough to offset variable selection incentive, hence double
                    # TODO: Check if double is needed, since any exceedance inherently violates multiple times anyways

        return qubo_dict

    def __add_position_constraints(
        self, constraint_factor: int, n: int
    ) -> dict[tuple[int, int], int]:
        """
        Strongly penalises revisiting the same node (aka vertical one-hot
        violations). When paired with time constraints, this inherently
        discourages more than n selections.

        Runs in O(n^3), creating n(n-1)^2 new edges.

        Parameters
        ----------
        constraint_factor : int
            Scaling factor to adjust constraint weighting.
        n : int
            Number of nodes in a route.

        Returns
        -------
        dict
            Collection of QUBO edge weights, indexed by 2-tuple of BV indices.
            Note that QUBO just sums (x, y) and (y, x) edges.
        """
        qubo_dict = {}

        # Same as time penalty, but downwards instead
        for row in range(n - 1):
            for col in range(n):
                qubit_a = row * n + col
                for row_offset in range(row + 1, n):
                    qubit_b = row_offset * n + col
                    qubo_dict[(qubit_a, qubit_b)] = 2 * constraint_factor

        return qubo_dict

    # Public methods
    def formulate(
        self,
        distance_matrix: np.ndarray,
        cost_factor: int,
        constraint_factor: int,
        is_circuit: bool,
    ) -> dict[tuple[int, int], int]:
        """
        Strongly penalises revisiting the same node (aka vertical one-hot
        violations). When paired with time constraints, this inherently
        discourages more than n selections.

        Runs in O(n^3), creating n(n-1)^2 new edges.

        Parameters
        ----------
        distance_matrix : ndarray
            2D asymmetric distance matrix.
        cost_factor : int
            Scaling factor to adjust distance weighting.
        constraint_factor : int
            Scaling factor to adjust constraint weighting.
        is_circuit : bool
            Toggle between circuit and path.

        Returns
        -------
        dict
            Collection of QUBO edge weights, indexed by 2-tuple of BV indices.
            Note that QUBO just sums (x, y) and (y, x) edges.
        """
        # Normalise, so that relative scaling factors work
        max_distance = np.max(np.array(distance_matrix))
        scaled_matrix = distance_matrix / max_distance
        n = len(distance_matrix)

        # Enforce minimum length
        cost_terms = self.__add_cost_objective(cost_factor, scaled_matrix, is_circuit)

        # Enforce Hamiltonian path
        incentive_terms = self.__add_selection_incentive(constraint_factor, n)
        time_terms = self.__add_time_constraints(constraint_factor, n)
        position_terms = self.__add_position_constraints(constraint_factor, n)

        return cost_terms | incentive_terms | time_terms | position_terms
