import pytest
import numpy as np
from route_optimisation.route_solver.brute_force_solver import BruteForceSolver

# Fixtures
@pytest.fixture
def dummy_bfs() -> BruteForceSolver:
    return BruteForceSolver()

@pytest.fixture
def simple_2x2() -> np.ndarray:
    # Simple 2x2 matrix
    return np.array([
        [0, 4],
        [2, 0]
    ])

@pytest.fixture
def simple_4x4() -> np.ndarray:
    # Simple 4x4 matrix
    return np.array([
        [0, 1, 100, 100],
        [100, 0, 1, 100],
        [100, 100, 0, 1],
        [100, 100, 100, 0]
    ])

@pytest.fixture
def equal_4x4() -> np.ndarray:
    # 4x4 matrix of equal values
    return np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ])

@pytest.fixture
def invalid_1x3() -> np.ndarray:
    # Invalid 1x3 matrix
    return np.array([
        [0, 4, 3]
    ])

@pytest.fixture
def invalid_3x1() -> np.ndarray:
    # Invalid 3x1 matrix
    return np.array([
        [0],
        [2],
        [3]
    ])

@pytest.fixture
def infinity_1x1() -> np.ndarray:
    # 1x1 matrix containing infinity
    return np.array([
        [float('inf')]
    ])

@pytest.fixture
def infinity_2x2() -> np.ndarray:
    # 2x2 matrix containing infinities
    return np.array([
        [float('inf'), float('inf')],
        [float('inf'), float('inf')]
    ])

@pytest.fixture
def normal_2x2() -> np.ndarray:
    # 2x2 matrix with random values
    return np.array([
        [10.52, 2.521],
        [3.82, 4.525]
    ])

@pytest.fixture
def normal_3x3() -> np.ndarray:
    # 3x3 matrix with random values
    return np.array([
        [0, 10.52, 2],
        [20.252, 0, 6.242],
        [1, 24, 0]
    ])

@pytest.fixture
def normal_4x4() -> np.ndarray:
    # 4x4 matrix with random values
    return np.array([
        [0, 10.52, 2, 42],
        [20.252, 0, 6.242, 953],
        [1, 24, 0, 1],
        [1.1, 24, 0, 25.2],
    ])

# Tests
def test_input_types(dummy_bfs : BruteForceSolver,
                       invalid_1x3 : np.ndarray,
                       invalid_3x1 : np.ndarray,
                       normal_3x3 : np.ndarray):
    """
    Test with inputs of differing types:
        None
        float
        str, size 1
        str, size 2
        1x3 matrix
        3x1 matrix
        3x3 matrix of negative values
    """
    # Test none
    with pytest.raises(TypeError):
        dummy_bfs.solve(None)

    # Test with float
    with pytest.raises(TypeError):
        dummy_bfs.solve(5.24)

    # Test with str, size 1
    assert dummy_bfs.solve('a') == ([0], 0)

    # Test with str, size 2
    with pytest.raises(IndexError):
        dummy_bfs.solve('ab')

    # Test with 1x3 matrix
    assert dummy_bfs.solve(invalid_1x3) == ([0], 0)

    # Test with 3x1 matrix
    with pytest.raises(IndexError):
        dummy_bfs.solve(invalid_3x1)

    # Test with 3x3 matrix of negative value
    assert dummy_bfs.solve(-normal_3x3) == ([2,1,0], pytest.approx(-44.252))


def test_first_route(dummy_bfs : BruteForceSolver,
                       equal_4x4 : np.ndarray):
    """
    Test that route solver stores first calculated route
        4x4 matrix of equal values
        Transpose of 4x4 matrix of equal values
    """
    # Test with 4x4 matrix of equal values
    assert dummy_bfs.solve(equal_4x4) == ([0,1,2,3], 3)

    # Test with transpose of 4x4 matrix of equal values
    equal_4x4_t = equal_4x4.T
    assert dummy_bfs.solve(equal_4x4_t) == ([0,1,2,3], 3)


def test_extreme_bounds(dummy_bfs : BruteForceSolver,
                        infinity_1x1 : np.ndarray,
                        infinity_2x2 : np.ndarray) -> None:
    """
    Test that route solver can deal with extreme values
        1x1 matrix containing infinity
        2x2 matrix containing infinities
        1x1 matrix containing negative infinity
        2x2 matrix containing negative infinities
    """
    # Test with 1x1 matrix containing infinity
    # BFS is not designed to handle 1x1 matrices, hence incorrect result
    assert dummy_bfs.solve(infinity_1x1) == ([0], 0)

    # Test with 2x2 matrix containing infinities
    # No valid route since infinity is not less than infinity
    assert dummy_bfs.solve(infinity_2x2) == ([], float('inf'))

    # Test with 1x1 matrix containing negative infinity
    # BFS is not designed to handle 1x1 matrices, hence incorrect result
    neg_infinity_1x1 = -infinity_1x1
    assert dummy_bfs.solve(neg_infinity_1x1) == ([0], 0)

    # Test with 2x2 matrix containing negative infinities
    # Picks first path
    neg_infinity_2x2 = -infinity_2x2
    assert dummy_bfs.solve(neg_infinity_2x2) == ([0, 1], -float('inf'))
    

def test_symmetrical_matrices(dummy_bfs : BruteForceSolver,
                              simple_2x2 : np.ndarray,
                              simple_4x4 : np.ndarray) -> None:
    """
    Test that if there is one unique optimal path, it is selected, despite
    being transposed
        2x2 simple matrix
        2x2 simple matrix, transposed
        4x4 simple matrix
        4x4 simple matrix, transposed
    """
    # 2x2 simple matrix
    route, cost = dummy_bfs.solve(simple_2x2)
    assert route == [1, 0] and cost == 2
    simple_2x2_t = simple_2x2.T

    # 2x2 simple matrix, transposed
    route_t, cost_t = dummy_bfs.solve(simple_2x2_t)
    assert route_t == route[::-1] and cost_t == cost

    # 4x4 simple matrix
    route, cost = dummy_bfs.solve(simple_4x4)
    assert route == [0, 1, 2, 3] and cost == 3
    simple_4x4_t = simple_4x4.T

    # 4x4 simple matrix, transposed
    route_t, cost_t = dummy_bfs.solve(simple_4x4_t)
    assert route_t == route[::-1] and cost_t == cost

def test_regular_matrices(dummy_bfs : BruteForceSolver,
                          normal_2x2 : np.ndarray,
                          normal_3x3 : np.ndarray,
                          normal_4x4 : np.ndarray) -> None:
    """
    Test different sizes of valid matrices
        1x1 matrix
        2x2 matrix
        3x3 matrix
        4x4 matrix
    """

    # 1x1 matrix
    # BFS is not designed to handle 1x1 matrices, hence incorrect result
    assert dummy_bfs.solve(np.array([1])) == ([0], 0)

    # 2x2 matrix
    assert dummy_bfs.solve(normal_2x2) == ([0,1], 2.521)
    
    # 3x3 matrix
    # 6.242 + 1
    assert dummy_bfs.solve(normal_3x3) == ([1, 2, 0], 7.242)

    # 4x4 matrix
    # 6.242 + 1 + 1.1
    assert dummy_bfs.solve(normal_4x4) == ([1, 2, 3, 0], 8.342)
