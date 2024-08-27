"""Factory for solving method"""

import os

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

from route_optimisation.route_solver.brute_force_solver import BruteForceSolver
from route_optimisation.route_solver.dwave_solver import DWaveSolver
from route_optimisation.route_solver.route_solver import RouteSolver


class SolverFactory:
    """
    Factory class for creating a particular SolverFactory instance
    """

    # Validate and decide what ATSP solver to build for the API
    def create(self, solver_type: str) -> RouteSolver:
        """
        Validate and decide what solving method to use for the API

        Parameters
        ----------
        solver_type : str
            solving method to be used

        Returns
        -------
        RouteSolver
            RouteSolver strategy depending on solver_type

        Raises
        ------
        ValueError
            If provided solver_type doesn't exist.
        """
        if solver_type == "dwave":
            # Max cluster size is already handled by the dwave solver
            # TODO: Might need to handle bad k on the endpoint to prevent 500s?

            # TODO: Params need deep tuning
            return DWaveSolver(
                EmbeddingComposite(
                    DWaveSampler(
                        token=os.environ["DWAVE_TOKEN"],
                        endpoint=os.environ["DWAVE_URL"],
                        solver=os.environ["DWAVE_SOLVER"],
                    )
                )
            )
            # Missing tokens are the app's (env) fault, so bubble up

        if solver_type == "brute":
            return BruteForceSolver()

        raise ValueError("Unsupported solver type.")
