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

import numpy

def mirror_across_plane(
    frame: Frame,
    normal: numpy.ndarray,
    point: numpy.ndarray,
    use_global: bool = False,
    inplace: bool = True,
) -> Frame:
    r"""
    Mirror a frame across a specified plane.

    The frame is mirrored across the plane defined by a normal vector and a point on the plane.

    If ``use_global`` is set to True, the mirroring is applied in the global coordinate system.
    Otherwise, the mirroring is applied in the local coordinate system of the frame (in the parent frame).

    .. figure:: ../../../py3dframe/resources/mirroring_across_plane.png
        :alt: Mirroring Across Plane
        :align: center

    .. warning::

        Mirroring a frame will change its handedness. If the original frame is right-handed, the mirrored frame will be left-handed, and vice versa.
        However :class:`py3dframe.Frame` only supports right-handed frames. Therefore, after mirroring, the x-axis of the output frame is flipped to restore right-handedness.

    .. seealso::

        - :class:`py3dframe.Frame` : for more information about the Frame class.
        - :func:`translate` : to translate a frame by given offsets along each axis.
        - :func:`rotate_around_axis` : to rotate a frame around a specific axis

    Parameters
    ----------
    frame : Frame
        The frame to mirror.

    normal : numpy.ndarray
        A 3D vector representing the normal of the plane across which to mirror the frame. The vector not need to be normalized.
        
    point : numpy.ndarray
        A 3D point on the plane across which to mirror the frame.

    use_global : bool, optional
        If True, the mirroring is applied in the global coordinate system.
        If False, the mirroring is applied in the local coordinate system of the frame.
        Default is False.

    inplace : bool, optional
        If True, the mirroring is applied to the input frame and the same frame is returned.
        If False, a new mirrored frame is returned and the input frame remains unchanged. Default is True.

    Returns
    -------
    Frame
        The mirrored frame (with x-axis flipped to restore right-handedness).

    
    Examples
    --------

    .. code-block:: python

        from py3dframe import Frame, mirror_across_plane
        import numpy as np

        # Create a default frame
        frame = Frame.canonical()

        # Define the normal of the plane (e.g., the xy-plane)
        normal = np.array([0.0, 0.0, 1.0])
        point = np.array([0.0, 0.0, 0.0])  # A point on the plane

        # Mirror the frame across the specified plane in the local coordinate system
        mirrored_frame = mirror_across_plane(frame, normal, point, use_global=False, inplace=True)

        # The orientation of the mirrored frame is now changed
        print(mirrored_frame.orientation)
    
    """
    if not isinstance(frame, Frame):
        raise TypeError("The 'frame' parameter must be an instance of Frame.")
    
    normal = numpy.asarray(normal, dtype=float).flatten()
    if normal.shape != (3,):
        raise ValueError("The 'normal' parameter must be a 3-dimensional vector.")
    if not numpy.isfinite(normal).all():
        raise ValueError("The 'normal' parameter must contain finite values.")
    
    norm = numpy.linalg.norm(normal)
    if numpy.abs(norm) < 1e-12:
        raise ValueError("The 'normal' parameter must be a non-zero vector.")
    
    normal_normalized = normal / norm

    point = numpy.asarray(point, dtype=float).flatten()
    if point.shape != (3,):
        raise ValueError("The 'point' parameter must be a 3-dimensional point.")
    if not numpy.isfinite(point).all():
        raise ValueError("The 'point' parameter must contain finite values.")
    
    if not inplace:
        frame = frame.copy()

    if use_global:
        frame_origin = frame.global_origin.flatten() 
        frame_axes = frame.global_axes
    else:
        frame_origin = frame.origin.flatten()
        frame_axes = frame.axes # Columns stack of x, y, z axes

    # Create the vectors to mirror
    origin_vector = frame_origin - point
    vectors = numpy.stack((origin_vector, frame_axes[:, 0], frame_axes[:, 1], frame_axes[:, 2]), axis=0)

    # Project the vectors onto the normal
    output_vectors = vectors - 2.0 * numpy.dot(vectors, normal_normalized)[:, numpy.newaxis] * normal_normalized
    mirrored_origin = point + output_vectors[0, :]
    mirrored_axes = numpy.stack((output_vectors[1, :], output_vectors[2, :], output_vectors[3, :]), axis=1)

    # Flip the x-axis to restore right-handedness
    mirrored_axes[:, 0] = -mirrored_axes[:, 0]

    if use_global:
        frame.global_origin = mirrored_origin
        frame.global_axes = mirrored_axes
    else:
        frame.origin = mirrored_origin
        frame.axes = mirrored_axes

    return frame