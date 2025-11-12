# Copyright 2025 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..frame import Frame
from numbers import Real
from typing import Optional

import numpy

def rotate_around_axis(
    frame: Frame,
    axis: numpy.ndarray,
    angle: float,
    point: Optional[numpy.ndarray] = None,
    use_global: bool = False,
    inplace: bool = True,
) -> Frame:
    r"""
    Rotate a frame around a specified axis by a given angle.

    The frame is rotated around the specified axis by the given angle in radians.

    If ``use_global`` is set to True, the rotation is applied in the global coordinate system.
    Otherwise, the rotation is applied in the local coordinate system of the frame (in the parent frame).

    .. figure:: ../../../py3dframe/resources/rotation_around_axis.png
        :alt: Rotation Around Axis
        :align: center

    .. seealso:: 

        - :class:`py3dframe.Frame` : for more information about the Frame class.
        - :func:`py3dframe.manipulations.translate` : to translate a frame by given offsets along each axis.
        - :func:`py3dframe.manipulations.translate_along_axis` : to translate a frame along a specific axis.

    Parameters
    ----------
    frame : Frame
        The frame to rotate.

    axis : numpy.ndarray
        A 3D vector representing the axis around which to rotate the frame. The vector not need to be normalized.

    angle : float
        The angle in radians to rotate around the specified axis.

    point : Optional[numpy.ndarray], optional
        A 3D point representing a point through which the rotation axis passes.
        If None, the rotation is performed around the frame's origin. Default is None.

    use_global : bool, optional
        If True, the rotation is applied in the global coordinate system.
        If False, the rotation is applied in the local coordinate system of the frame.
        Default is False.

    inplace : bool, optional
        If True, the rotation is applied to the input frame and the same frame is returned.
        If False, a new rotated frame is returned and the input frame remains unchanged. Default is True.

    Returns
    -------
    Frame
        The rotated frame.

    
    Examples
    --------

    .. code-block:: python

        from py3dframe import Frame
        from py3dframe.manipulations import rotate_around_axis
        import numpy as np

        # Create a default frame
        frame = Frame.canonical()

        # Define an axis to rotate around (e.g., the z-axis)
        axis = np.array([0.0, 0.0, 1.0])
        angle_rad = np.pi / 2  # 90 degrees in radians

        # Rotate the frame by 90 degrees around the specified axis in the local coordinate system
        rotated_frame = rotate_around_axis(frame, axis, angle_rad, use_global=False, inplace=True)

        # The orientation of the rotated frame is now changed
        print(rotated_frame.orientation)

    """
    if not isinstance(frame, Frame):
        raise TypeError("The 'frame' parameter must be an instance of Frame.")
    if not isinstance(angle, Real):
        raise TypeError("The 'angle' parameter must be a real number.")
    if not numpy.isfinite(angle):
        raise ValueError("The 'angle' parameter must be finite.")
    
    axis = numpy.asarray(axis, dtype=float).flatten()
    if axis.shape != (3,):
        raise ValueError("The 'axis' parameter must be a 3-dimensional vector.")
    if not numpy.isfinite(axis).all():
        raise ValueError("The 'axis' parameter must contain finite values.")
    
    norm = numpy.linalg.norm(axis)
    if numpy.abs(norm) < 1e-12:
        raise ValueError("The 'axis' parameter must be a non-zero vector.")
    
    axis_normalized = axis / norm

    if point is not None:
        point = numpy.asarray(point, dtype=float).flatten()
        if point.shape != (3,):
            raise ValueError("The 'point' parameter must be a 3-dimensional point.")
        if not numpy.isfinite(point).all():
            raise ValueError("The 'point' parameter must contain finite values.")
    else:
        point = frame.origin.flatten() if not use_global else frame.global_origin.flatten()

    if not inplace:
        frame = frame.copy()

    if use_global:
        frame_origin = frame.global_origin.flatten() # (3,)
        frame_axes = frame.global_axes # Columns stack of x, y, z axes
    else:
        frame_origin = frame.origin.flatten() # (3,)
        frame_axes = frame.axes # Columns stack of x, y, z axes

    # Create the inputs points
    input_origin = frame_origin # (3,)
    input_x = frame_origin + frame_axes[:, 0] # (3,)
    input_y = frame_origin + frame_axes[:, 1] # (3,)
    input_z = frame_origin + frame_axes[:, 2] # (3,)
    input_points = numpy.stack((input_origin, input_x, input_y, input_z), axis=0)

    # Rotate the points around the axis
    vectors = input_points - point
    parallel_components = numpy.dot(vectors, axis_normalized)[:, numpy.newaxis] * axis_normalized
    perpendicular_components = vectors - parallel_components

    # Create local x and y axes with respect to the rotation axis
    min_index = numpy.argmin(numpy.abs(axis_normalized))
    if min_index == 0:
        temp_vector = numpy.array([1.0, 0.0, 0.0])
    elif min_index == 1:
        temp_vector = numpy.array([0.0, 1.0, 0.0])
    else:
        temp_vector = numpy.array([0.0, 0.0, 1.0])
    local_x = numpy.cross(axis_normalized, temp_vector)
    local_x /= numpy.linalg.norm(local_x)
    local_y = numpy.cross(axis_normalized, local_x) 
    local_y /= numpy.linalg.norm(local_y)

    # Create the projection of the orthogonal components onto the local x and y axes
    x_proj = numpy.dot(perpendicular_components, local_x)
    y_proj = numpy.dot(perpendicular_components, local_y)

    # Rotate the local_x and local_y
    x_rotated = local_x * numpy.cos(angle) + local_y * numpy.sin(angle)
    y_rotated = -local_x * numpy.sin(angle) + local_y * numpy.cos(angle)

    rotated_perpendicular_components = x_proj[:, numpy.newaxis] * x_rotated + y_proj[:, numpy.newaxis] * y_rotated
    rotated_vectors = parallel_components + rotated_perpendicular_components
    rotated_points = point + rotated_vectors

    # Update the frame with the rotated points
    rotated_origin = rotated_points[0, :]
    rotated_x = rotated_points[1, :] - rotated_origin
    rotated_y = rotated_points[2, :] - rotated_origin
    rotated_z = rotated_points[3, :] - rotated_origin
    rotated_axes = numpy.stack((rotated_x, rotated_y, rotated_z), axis=1)

    if use_global:
        frame.global_origin = rotated_origin
        frame.global_axes = rotated_axes
    else:
        frame.origin = rotated_origin
        frame.axes = rotated_axes
    
    return frame

