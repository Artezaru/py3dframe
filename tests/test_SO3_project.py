import numpy as np
import pytest
from py3dframe.matrix import SO3_project, is_SO3

def test_SO3_project_valid_matrix():
    """Test the projection onto SO(3) for a valid matrix."""
    
    # Case: initial matrix
    e1 = np.array([1, 1, 0])
    e2 = np.array([-1, 1, 0])
    e3 = np.array([0, 0, 1])
    matrix = np.column_stack((e1, e2, e3))

    expected_matrix =  matrix / np.linalg.norm(matrix, axis=0)
    
    # Projected onto SO(3)
    projected_matrix = SO3_project(matrix)
    
    # Check orthogonality using is_SO3
    assert is_SO3(projected_matrix), "The projected matrix is not orthogonal"
    
    # Check the shape (3x3)
    assert projected_matrix.shape == (3, 3), f"The shape of the projected matrix is incorrect: {projected_matrix.shape}"
    
    # Check equality with the expected matrix
    assert np.allclose(projected_matrix, expected_matrix), "The projected matrix is incorrect"

def test_SO3_project_random_matrix():
    """Test the projection of a random matrix onto SO(3)."""
    
    # Generate a random 3x3 matrix
    matrix = np.random.rand(3, 3)
    
    # Projected onto SO(3)
    projected_matrix = SO3_project(matrix)
    
    # Check orthogonality using is_SO3
    assert is_SO3(projected_matrix), "The projected matrix is not orthogonal"
    
    # Check the determinant
    det = np.linalg.det(projected_matrix)
    assert np.isclose(det, 1), f"The determinant is {det}, it should be 1"

def test_SO3_project_identity_matrix():
    """Test the projection onto SO(3) for the identity matrix."""
    
    # Case: identity matrix
    matrix = np.eye(3)
    
    # Projected onto SO(3)
    projected_matrix = SO3_project(matrix)
    
    # Check that the matrix remains unchanged after projection
    assert np.allclose(projected_matrix, matrix), "The projected matrix is not equal to the identity matrix"
    
    # Check the determinant
    det = np.linalg.det(projected_matrix)
    assert np.isclose(det, 1), f"The determinant is {det}, it should be 1"
    
    # Check orthogonality using is_SO3
    assert is_SO3(projected_matrix), "The projected matrix is not orthogonal"
