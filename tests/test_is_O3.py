import numpy as np
import pytest
from py3dframe.matrix import is_O3

def test_is_O3_valid_matrices():
    """ Test valid orthonormal matrices. """

    # Test case where matrix is orthonormal (should return True)
    e1 = np.array([1, 1, 0]) / np.sqrt(2)
    e2 = np.array([-1, 1, 0]) / np.sqrt(2)
    e3 = np.array([0, 0, 1])
    matrix = np.column_stack((e1, e2, e3))
    
    assert is_O3(matrix), "Matrix should be in the orthogonal group O(3)"

    # Another valid orthonormal matrix
    e1 = np.array([-1, 1, 0]) / np.sqrt(2)
    e2 = np.array([1, 1, 0]) / np.sqrt(2)
    e3 = np.array([0, 0, 1])
    matrix = np.column_stack((e1, e2, e3))
    
    assert is_O3(matrix), "Matrix should be in the orthogonal group O(3)"

def test_is_O3_invalid_matrices():
    """ Test invalid orthonormal matrices. """

    # Non-normalized matrix
    e1 = np.array([1, 1, 1])
    e2 = np.array([-1, 1, 0])
    e3 = np.array([0, 0, 1])
    matrix = np.column_stack((e1, e2, e3))

    assert not is_O3(matrix), "Matrix should not be in the orthogonal group O(3)"
    
    # Non-orthonormal matrix
    e1 = np.array([1, 1, 1]) / np.sqrt(3)
    e2 = np.array([-1, 1, 0]) / np.sqrt(2)
    e3 = np.array([0, 0, 1])
    matrix = np.column_stack((e1, e2, e3))
    
    assert not is_O3(matrix), "Matrix should not be in the orthogonal group O(3)"

    # Non-3x3 matrix
    matrix = np.array([1, 2, 3, 4]).reshape(2, 2)
    
    with pytest.raises(ValueError):
        is_O3(matrix)

def test_is_O3_tolerance():
    """ Test with a tolerance value. """
    
    # Slightly perturbed orthonormal matrix
    e1 = np.array([1, 1, 0]) / np.sqrt(2)
    e2 = np.array([-1, 1, 0]) / np.sqrt(2)
    e3 = np.array([0, 0, 1])
    matrix = np.column_stack((e1, e2, e3))
    
    # Perturb the matrix a little
    matrix[0, 0] += 1e-7  # Slight perturbation
    
    assert is_O3(matrix, tolerance=1e-6), "Matrix should still be considered orthonormal with given tolerance"
    
    # Larger perturbation
    matrix[0, 0] += 1e-3  # Larger perturbation
    
    assert not is_O3(matrix, tolerance=1e-6), "Matrix should no longer be considered orthonormal with larger perturbation"
