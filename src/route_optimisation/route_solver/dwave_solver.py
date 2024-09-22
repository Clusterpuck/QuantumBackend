import numpy as np
import dimod

from .route_solver import RouteSolver
from .quantum_tsp.tsp_formulation import TSPFormulation


class DWaveSolver(RouteSolver):

    def __init__(
        self,
        sampler: dimod.Sampler,  # Injection allows mocking for offline testing
        cost_factor: int = 10,
        constraint_factor: int = 800,
        chain_strength: int = 1600,
        num_runs: int = 1024,
        max_retries: int = 3,
        is_circuit: bool = False,
    ):
        """
        Inits an ATSP solver that uses D-Wave machines.

        Cost-to-constraint ratio of 80 at chain strength 1600 has been
        experimentally verified to perform decently for <=7 nodes with current
        technology (Advantage 1) on paths. >=8 are generally unsolvable or inconsistent.

        1024 runs per call gives stable results, but it's possible to reduce
        this to as low as 200 at the risk of inconsistent solution quality or
        even complete failures.

        Parameters
        ----------
        sampler : Sampler
            Configured, DWave sampler.
        cost_factor : int, default=10
            Scales QUBO weight for costs. Cannot exceed constraint factor.
        constraint_factor : int, default=800
            Scales QUBO weight for penalising constraint violations.
        chain_strength : int, default=800
            Decreases likelihood of chain breaks, but drifts from original problem definition.
        num_runs : int, default=1024
            More runs increase the probability of finding the best solution,
            but may hit D-Wave's license cap or per-call runtime limit.
        max_retries : int, default=3
            Max reattempts if a valid route could not be found.
        is_circuit : bool, default=False
            If False, reformulates as a minimum path variation.
        """
        self.__sampler = sampler
        self.__cost_factor = cost_factor
        self.__constraint_factor = constraint_factor
        self.__chain_strength = chain_strength
        self.__num_runs = num_runs
        self.__max_retries = max_retries
        self.__is_circuit = is_circuit

    def solve(self, distance_matrix: np.ndarray) -> tuple[list[int], int]:
        """
        Compute a good solution for TSP via D-Wave.

        Parameters
        ----------
        distance_matrix : ndarray
            Asymmetric distance matrix to find a shortest route for.

        Returns
        -------
        best_route : list of int
            The valid route with the lowest energy.
        int
            Distance if the route is followed.

        Raises
        ------
        ValueError
            If distance matrix is invalid or clearly oversized.
        RuntimeError
            If D-Wave cannot find a valid solution within the number of retries
        """
        # Fast fail
        if len(distance_matrix) < 2:
            raise ValueError("Distance matrix must be 2x2 or larger.")
        elif distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("Distance matrix must be square.")
        elif len(distance_matrix) > 10:
            raise ValueError("D-Wave cannot handle more than 10 TSP nodes.")

        # Create QUBO
        qubo_formulator = TSPFormulation()
        qubo = qubo_formulator.formulate(
            distance_matrix,
            self.__cost_factor,
            self.__constraint_factor,
            self.__is_circuit,
        )

        # Try sending to D-Wave machine
        tries = 0
        best_route = None
        while best_route is None and tries < self.__max_retries:
            response = self.__sampler.sample_qubo(
                qubo, chain_strength=self.__chain_strength, num_reads=self.__num_runs
            )
            best_route = self.__decode_solution(response)
            tries += 1

        if best_route is None:
            raise RuntimeError("No valid D-Wave solution received")

        # Finally, derive cost
        current_cost = self.__get_route_cost(
            best_route, distance_matrix, self.__is_circuit
        )

        return best_route, current_cost

    def __get_route_cost(
        self, route: list[int], distance_matrix: np.ndarray, is_circuit: bool
    ) -> int:
        """
        Calculate the cost of a route solution.

        Parameters
        ----------
        route : list of int or None
            Complete, indexed locations in visit order.
        distance_matrix : ndarray
            Data to compute the total distance with.
        is_circuit : bool
            Whether to include the loop back to start.

        Returns
        -------
        int
            Distance if the route is followed.
        """
        cost = 0
        for i in range(len(route) - 1):
            cost += distance_matrix[route[i], route[i + 1]]

        # Loop around if circuit
        if is_circuit:
            cost += distance_matrix[route[-1], route[0]]

        return cost

    def __validate_permutation(self, matrix: np.ndarray) -> bool:
        """
        Validate that permutation matrix is correctly one-hot encoded.

        Parameters
        ----------
        matrix : ndarray
            Permutation matrix. Assumes binary matrix.

        Returns
        -------
        bool
            True if one-hot encoded.
        """
        # Every row/col sums to 1
        return np.all(np.sum(matrix, axis=0) == 1) and np.all(
            np.sum(matrix, axis=1) == 1
        )

    def __decode_solution(self, response: dimod.SampleSet) -> list[int] | None:
        """
        Decodes BV permutation matrix, returning 0-based route.

        Parameters
        ----------
        response : SampleSet
            Response from D-Wave (QUBO-specific). Contains sampled info in a
            NumPy recarray format. See official docs: https://docs.ocean.dwavesys.com/en/stable/docs_dimod/reference/sampleset.html

        Returns
        -------
        best_solution : list of int or None
            The valid route with the lowest energy, if found.
        """
        valid_distribution = {}
        min_energy = np.max(response.record.energy)
        best_solution = None
        node_count = int(np.sqrt(len(response.record[0].sample)))

        for entry in response.record:
            solution_matrix = np.reshape(entry.sample, (node_count, node_count))

            # Filter to only valid ones
            if self.__validate_permutation(solution_matrix):
                # Convert to python ints for consistent strategy interface
                solution = [int(np.where(row == 1)[0][0]) for row in solution_matrix]
                valid_distribution[tuple(solution)] = (
                    entry.energy,
                    entry.num_occurrences,
                )
                # Track the best one
                if entry.energy <= min_energy:
                    min_energy = entry.energy
                    best_solution = solution

        return best_solution
