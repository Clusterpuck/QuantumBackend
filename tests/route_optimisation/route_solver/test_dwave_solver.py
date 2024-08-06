from typing import Any
import dimod
import numpy as np
import pytest
import sys

from route_optimisation.route_solver.dwave_solver import DWaveSolver


class MockSampler(dimod.Sampler):
    # Don't bother implementing abstracts
    def parameters(self): pass
    def properties(self): pass

    def sample_qubo(
        self, Q: dict[tuple[int, int], Any], **parameters
    ) -> dimod.SampleSet:
        # Ignore inputs, just simulate for 4 cities (16 BVs)

        # Only one-hot encoded BVs are valid
        valid_sample = [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        # Corresponds to route [3, 2, 0, 1]

        # Pretend to measure 5 samples
        samples = [
            [0] * 16,
            [1] * 16,  # Lowest energy, but invalid
            [0] * 16,
            valid_sample,  # Lowest valid
            [0] * 16,
        ]
        energies = [5, -300, 120, 30, -30]

        return dimod.SampleSet.from_samples(samples, "BINARY", energies)


class MockUnstableSampler(dimod.Sampler):
    def parameters(self): pass
    def properties(self): pass

    def __init__(self, max_retries):
        self.__call_count = 0
        self.__max_retries = max_retries

    def sample_qubo(
        self, Q: dict[tuple[int, int], Any], **parameters
    ) -> dimod.SampleSet:
        # Same as MockSampler, succeeds only on the last try (of max_retries)

        if self.__call_count < self.__max_retries - 1:
            self.__call_count += 1
            sample_set = dimod.SampleSet.from_samples([[0] * 16] * 5, "BINARY", 0)
        else:
            valid_sample = [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0]
            samples = [
                [0] * 16,
                [1] * 16,
                [0] * 16,
                valid_sample,
                [0] * 16,
            ]
            energies = [5, -300, 12, 30, -30]
            sample_set = dimod.SampleSet.from_samples(samples, "BINARY", energies)

        return sample_set


@pytest.fixture
def dummy_asym_matrix():
    # Solvers using MockSampler should return [3, 2, 0, 1] as the best route
    # Decoding it using this distance matrix gives a cost of 4 (path)
    return np.array(
        [
            [0, 1, 100, 100],
            [100, 0, 100, 1],
            [1, 100, 0, 100],
            [100, 100, 2, 0],
        ]
    )


# @pytest.fixture
# def asym_matrix_4x4():
#     return np.array(
#         [
#             [0, 50, 1, 82],
#             [1, 0, 40, 76],
#             [12, 23, 0, 1],
#             [99, 2, 94, 0],
#         ]
#     )
#     # Path should find [1, 0, 2, 3], 1 + 1 + 1 = 3
#     # Circuit should find some shift of [2, 3, 1, 0], 1 + 2 + 1 + 1 = 5


# @pytest.fixture
# def asym_matrix_5x5():
#     # Don't use ExactSolver for this. Computing all energies takes a while.
#     return np.array(
#         [
#             [0, 50, 75, 82, 1],
#             [1, 0, 40, 76, 32],
#             [12, 23, 0, 1, 23],
#             [99, 2, 94, 0, 62],
#             [24, 67, 1, 75, 0],
#         ]
#     )
#     # Path should find [0, 4, 2, 3], 1 + 1 + 1 + 1 = 5
#     # Circuit should find some shift of [0, 4, 2, 3, 1], 1 + 1 + 1 + 1 + 2 = 6


@pytest.fixture
def dummy_response():  # TODO: For testing private __decode_solution
    # Imitates what sample_qubo would send back
    return dimod.SampleSet.from_samples(np.ones(5, dtype="int8"), "BINARY", 0)


def test_solve(dummy_asym_matrix):
    # Only test that decoding and retries work
    # QUBO correctness can be tested in a tsp formulation test
    # Sampler args can be tested in a main test

    # Test success and decoding
    dummy_solver = DWaveSolver(MockSampler())
    expected = ([3, 2, 0, 1], 4)
    result = dummy_solver.solve(dummy_asym_matrix)
    assert expected == result

    # Test retries. Succeeds on 3rd try, barely within default retry cap
    dummy_solver = DWaveSolver(MockUnstableSampler(3))
    expected = ([3, 2, 0, 1], 4)
    result = dummy_solver.solve(dummy_asym_matrix)
    assert expected == result

    # These should fail
    with pytest.raises(RuntimeError):
        dummy_solver = DWaveSolver(MockUnstableSampler(4))
        dummy_solver.solve(dummy_asym_matrix)

    with pytest.raises(RuntimeError):
        dummy_solver = DWaveSolver(MockUnstableSampler(2), max_retries=1)
        dummy_solver.solve(dummy_asym_matrix)

# TODO: Since the private methods are bulky, test those too
# With extra time, maybe also test alt features in case they are used later
