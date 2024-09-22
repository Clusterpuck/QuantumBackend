from typing import Any
import dimod
import numpy as np
import pytest

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
        valid_sample = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
        # Corresponds to route [2, 0, 1, 3]

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
            valid_sample = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
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
def dummy_asym_matrix() -> np.ndarray:
    # Solvers using MockSampler should return [2, 0, 1, 3] as the best route
    # Due to the fake sampler, this is only used for decoding the expected
    # cost of 3 (path).
    return np.array(
        [
            [0, 1, 100, 100],
            [100, 0, 100, 1],
            [1, 100, 0, 100],
            [100, 100, 2, 0],
        ]
    )


@pytest.fixture
def dummy_single_response() -> dimod.SampleSet:
    # Imitates what sample_qubo would send back
    # If single valid, that should be only answer
    samples = [[0, 0, 1, 0, 1, 0, 1, 0, 0]]
    energies = [5]
    return dimod.SampleSet.from_samples(samples, "BINARY", energies)

@pytest.fixture
def dummy_many_response() -> dimod.SampleSet:
    # If many valid, it should pick the best
    samples = [
        [0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],  # Lowest energy valid
    ]
    energies = [1, 2, -10]
    return dimod.SampleSet.from_samples(samples, "BINARY", energies)

@pytest.fixture
def dummy_mixed_response() -> dimod.SampleSet:
    # If mixed, select the best valid
    samples = [
        [0] * 9,
        [1] * 9,
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 0],  # Lowest energy valid
    ]
    energies = [-100, -100, 5, 4]
    return dimod.SampleSet.from_samples(samples, "BINARY", energies)

@pytest.fixture
def dummy_invalid_response() -> dimod.SampleSet:
    return dimod.SampleSet.from_samples(np.ones(9, dtype="int8"), "BINARY", 0)


def test_solve(dummy_asym_matrix):
    # Only test that decoding and retries work
    # QUBO correctness can be tested in a tsp formulation test
    # Sampler args can be tested in a main test

    # Test success and decoding
    dummy_solver = DWaveSolver(MockSampler())
    expected = ([2, 0, 1, 3], 3)
    result = dummy_solver.solve(dummy_asym_matrix)
    assert expected == result

    # Test retrying succeeds on 3rd try, barely within default retry cap
    dummy_solver = DWaveSolver(MockUnstableSampler(3))
    expected = ([2, 0, 1, 3], 3)
    result = dummy_solver.solve(dummy_asym_matrix)
    assert expected == result

    # Bounds check the other side of the retry limit
    with pytest.raises(RuntimeError):
        dummy_solver = DWaveSolver(MockUnstableSampler(4))
        dummy_solver.solve(dummy_asym_matrix)

    with pytest.raises(RuntimeError):
        dummy_solver = DWaveSolver(MockUnstableSampler(2), max_retries=1)
        dummy_solver.solve(dummy_asym_matrix)

def test_get_route_cost(dummy_asym_matrix: np.ndarray):
    dummy_solver = DWaveSolver(MockSampler())

    # Test cost calculation when assuming (realistic) path
    result = dummy_solver._DWaveSolver__get_route_cost([0, 1, 2, 3], dummy_asym_matrix, False)
    assert result == 201  # 1 + 100 + 100
    result = dummy_solver._DWaveSolver__get_route_cost([2, 0, 1, 3], dummy_asym_matrix, False)
    assert result == 3  # 1 + 1 + 1

    # Test cost calculation when assuming (realistic) circuit
    result = dummy_solver._DWaveSolver__get_route_cost([0, 1, 2, 3], dummy_asym_matrix, True)
    assert result == 301  # 1 + 100 + 100 + 100
    result = dummy_solver._DWaveSolver__get_route_cost([2, 0, 1, 3], dummy_asym_matrix, True)
    assert result == 5  # 1 + 1 + 1 + 2

def test_validate_permutation():
    dummy_solver = DWaveSolver(MockSampler())

    # Test one-hot is valid
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert dummy_solver._DWaveSolver__validate_permutation(matrix)
    matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert dummy_solver._DWaveSolver__validate_permutation(matrix)

    # Test various invalid cases
    matrix = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    assert not dummy_solver._DWaveSolver__validate_permutation(matrix)
    matrix = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    assert not dummy_solver._DWaveSolver__validate_permutation(matrix)
    matrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert not dummy_solver._DWaveSolver__validate_permutation(matrix)
    matrix = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    assert not dummy_solver._DWaveSolver__validate_permutation(matrix)

    # Obviously assuming binary int, so testing negatives etc is out of scope

def test_decode_solution(
    dummy_single_response: dimod.SampleSet,
    dummy_many_response: dimod.SampleSet,
    dummy_mixed_response: dimod.SampleSet,
    dummy_invalid_response: dimod.SampleSet
):
    dummy_solver = DWaveSolver(MockSampler())

    # Test that it always finds the lowest energy valid answer
    result = dummy_solver._DWaveSolver__decode_solution(dummy_single_response)
    assert result == [2, 1, 0]
    result = dummy_solver._DWaveSolver__decode_solution(dummy_many_response)
    assert result == [0, 1, 2]
    result = dummy_solver._DWaveSolver__decode_solution(dummy_mixed_response)
    assert result == [2, 1, 0]

    # Test that no valid solution returns None
    result = dummy_solver._DWaveSolver__decode_solution(dummy_invalid_response)
    assert result is None
