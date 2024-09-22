from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import os

from route_optimisation.route_solver.brute_force_solver import BruteForceSolver
from route_optimisation.route_solver.dwave_solver import DWaveSolver
from route_optimisation.route_solver.route_solver import RouteSolver


class SolverFactory:
    # Validate and decide what ATSP solver to build for the API
    def create(self, solver_type: str) -> RouteSolver:
        if solver_type == "dwave":
            return DWaveSolver(
                EmbeddingComposite(
                    DWaveSampler(
                        token=os.environ["DWAVE_TOKEN"],
                        endpoint=os.environ["DWAVE_URL"],
                        solver=os.environ["DWAVE_SOLVER"],
                    )
                )
            )
        # Missing tokens are the app's (env var) fault, so bubble up
        # NOTE: Solver auto fails fast on >10 nodes, since current machines
        # simply can't handle it

        elif solver_type == "brute":
            return BruteForceSolver()

        else:
            raise ValueError("Unsupported solver type.")
