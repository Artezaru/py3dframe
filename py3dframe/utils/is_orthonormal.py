import numpy
from typing import Optional

def is_orthonormal(rotation_matrix: numpy.ndarray, tolerance: Optional[float] = 1e-6) -> bool:
    """
    Check if the given rotation matrix is orthonormal.

    A matrix is orthonormal if its transpose is equal to its inverse.

    .. math::

        R^T = R^{-1}

    Parameters
    ----------
    rotation_matrix : numpy.ndarray
        The rotation matrix with shape (3, 3).

    tolerance : float, optional
        The tolerance for the comparison of the matrix with the identity matrix.
    
    Returns
    -------
    bool
        True if the matrix is orthonormal, False otherwise.

    Examples
    --------
    >>> import numpy
    >>> from py3dframe.utils import is_orthonormal
    >>> rotation_matrix = numpy.eye(3)
    >>> print(is_orthonormal(rotation_matrix))
    True
    """
    # Check if the matrix is square.
    if rotation_matrix.shape != (3, 3):
        raise ValueError("The rotation matrix must be 3x3.")
    
    # Check if the matrix is orthogonal.
    if not numpy.allclose(numpy.dot(rotation_matrix, rotation_matrix.T), numpy.eye(3), atol=tolerance):
        return False
    
    return True