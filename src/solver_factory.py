"""Validate and decide what ATSP solver to build for the API"""

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import os

from route_optimisation.route_solver.brute_force_solver import BruteForceSolver
from route_optimisation.route_solver.dwave_solver import DWaveSolver
from route_optimisation.route_solver.route_solver import RouteSolver


class SolverFactory:
    """
    Validate and decide what ATSP solver to build for the API
    """

    def create(self, solver_type: str) -> RouteSolver:
        """
        Creates ATSP solver based on provided solver type

        Parameters
        ----------
        solver_type: str
            Type of ATSP solver

        Returns
        -------
        RouteSolver
            Specific ATSP solver object

        Raises
        ------
        ValueError
            If solver type is unknown
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

        elif solver_type == "brute":
            return BruteForceSolver()

        else:
            raise ValueError("Unsupported solver type.")
