import numpy
from typing import Optional
from ..frame_old import Frame
from .is_orthonormal import is_orthonormal

def orthonormal_frame(
    origin: Optional[numpy.ndarray] = None,
    x_axis: Optional[numpy.ndarray] = None,
    y_axis: Optional[numpy.ndarray] = None,
    z_axis: Optional[numpy.ndarray] = None,
    solve_orthogonality: bool = False,
    tolerance: Optional[float] = 1e-6,
) -> numpy.ndarray:
    r"""
    Return an orthonormal frame of reference from the given axes.

    If no axes are given, the function will return the standard frame.

    If only one axis is given, the function will search for the best
    candidate for the others axes.

    If two axes are given, the function will compute the last axis.
    Then (if ``solve_orthogonality`` is True), the function will check if
    the axes are orthogonal. If not, the function will correct the axes 
    to be orthogonal. The second axis is the cross product of the first and
    third axes. In fact, the plane defined by the first and second axes
    is consistent.

    If all axes are given, the function will check if the axes are
    orthogonal. If not (and ``solve_orthogonality`` is True), the function
    will correct the axes to be orthogonal. The third axis is the cross
    product of the first and second axes and then the second axis is the
    cross product of the third and first axes. In fact, the plane defined
    by the first and second axes is consistent.

    .. warning::

        If ``solve_orthogonality`` is True, no message will be displayed if
        the axes are not orthogonal. 

    Parameters
    ----------
    origin : numpy.ndarray, optional
        The origin of the frame with shape (3,).

    x_axis : numpy.ndarray, optional
        The x-axis of the basis with shape (3,).
    
    y_axis : numpy.ndarray, optional
        The y-axis of the basis with shape (3,).

    z_axis : numpy.ndarray, optional
        The z-axis of the basis with shape (3,).

    solve_orthogonality : bool, optional
        If True, the function will correct the axes to be orthogonal.

    tolerance : float, optional
        The tolerance for the criteria of orthogonality.
    
    Returns
    -------
    Frame
        The orthonormal frame of reference.

    Raises  
    ------
    ValueError
        If a given axis has 0 magnitude.
        If the given axes are colinear.

    Examples
    --------
    >>> import numpy
    >>> from py3dframe.utils import orthonormal_frame
    >>> origin = numpy.array([0, 0, 0])
    >>> x_axis = numpy.array([1, 0, 0])
    >>> y_axis = numpy.array([0, 1, 0])
    >>> frame = orthonormal_frame(origin, x_axis=x_axis, y_axis=y_axis)
    """
    def normalize(axis: numpy.ndarray) -> None:
        axis = numpy.array(axis, dtype=numpy.float32).reshape((3,))
        norm = numpy.linalg.norm(axis)
        if numpy.isclose(norm, 0, atol=tolerance):
            raise ValueError("The given axis has 0 magnitude.")
        return axis / norm

    # If the origin is not given, set it to the origin.
    if origin is None:
        origin = numpy.zeros(3)
    origin = numpy.array(origin, dtype=numpy.float32).reshape((3,))

    # Case 0. No axes are given.
    if x_axis is None and y_axis is None and z_axis is None:
        x_axis = numpy.array([1, 0, 0])
        y_axis = numpy.array([0, 1, 0])
        z_axis = numpy.array([0, 0, 1])



    # Case 2. Only one axis is given.
    elif sum([x_axis is not None, y_axis is not None, z_axis is not None]) == 1:
        # Get the given axis.
        if x_axis is not None:
            first_axis = x_axis
        elif y_axis is not None:
            first_axis = y_axis
        else:
            first_axis = z_axis

        # Prepare the axis.
        first_axis = normalize(first_axis)

        # Search for the best candidate for the second axis.
        candidates = [
            numpy.array([first_axis[1], -first_axis[0], 0]),
            numpy.array([first_axis[2], 0, -first_axis[0]]),
            numpy.array([0, first_axis[2], -first_axis[1]]),
        ]

        candidates_norm = [numpy.linalg.norm(candidate) for candidate in candidates]
        second_axis = candidates[numpy.argmax(candidates_norm)]

        # Compute the third axis.
        third_axis = numpy.cross(first_axis, second_axis)

        # Return the axis.
        if x_axis is not None:
            x_axis = first_axis
            y_axis = second_axis
            z_axis = third_axis
        elif y_axis is not None:
            x_axis = third_axis
            y_axis = first_axis
            z_axis = second_axis
        elif z_axis is not None:
            x_axis = second_axis
            y_axis = third_axis
            z_axis = first_axis
    


    # Case 3. Two axes are given.
    elif sum([x_axis is not None, y_axis is not None, z_axis is not None]) == 2:
        # Get the first and second axes.
        if z_axis is None:
            first_axis = x_axis
            second_axis = y_axis
        elif y_axis is None:
            first_axis = z_axis
            second_axis = x_axis
        else:
            first_axis = y_axis
            second_axis = z_axis

        # Prepare the axes.
        first_axis = normalize(first_axis)
        second_axis = normalize(second_axis)

        # Check if the axes are colinear.
        if numpy.allclose(numpy.cross(first_axis, second_axis), 0, atol=tolerance):
            raise ValueError("The given axes are colinear.")
        
        # Compute the third axis.
        third_axis = numpy.cross(first_axis, second_axis)

        # Correct the axes to be orthogonal.
        if solve_orthogonality:
            second_axis = numpy.cross(third_axis, first_axis)
        
        # Return the axis.
        if x_axis is None:
            x_axis = third_axis
            y_axis = first_axis
            z_axis = second_axis
        elif y_axis is None:
            x_axis = second_axis
            y_axis = third_axis
            z_axis = first_axis
        elif z_axis is None:
            x_axis = first_axis
            y_axis = second_axis
            z_axis = third_axis
    


    # Case 4. All axes are given.
    else:
        # Prepare the axes.
        first_axis = normalize(x_axis)
        second_axis = normalize(y_axis)
        third_axis = normalize(z_axis)

        # Check if the axes are colinear.
        if numpy.allclose(numpy.cross(first_axis, second_axis), 0, atol=tolerance):
            raise ValueError("The given axes are colinear.")
        if numpy.allclose(numpy.cross(second_axis, third_axis), 0, atol=tolerance):
            raise ValueError("The given axes are colinear.")
        if numpy.allclose(numpy.cross(third_axis, first_axis), 0, atol=tolerance):
            raise ValueError("The given axes are colinear.")

        # Correct the axes to be orthogonal.
        if solve_orthogonality:
            third_axis = numpy.cross(first_axis, second_axis)
            second_axis = numpy.cross(third_axis, first_axis)
        
        # Return the axis.
        x_axis = first_axis
        y_axis = second_axis
        z_axis = third_axis
    
    # Return the frame.
    rotation_matrix = numpy.hstack((x_axis.reshape((3, 1)), y_axis.reshape((3, 1)), z_axis.reshape((3, 1))))
    if not is_orthonormal(rotation_matrix, tolerance):
        raise ValueError("The given axes are not orthogonal.")
    return Frame(origin=origin, rotation_matrix=rotation_matrix)


        


