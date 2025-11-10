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

def translate_along_axis(
    frame: Frame,
    axis: numpy.ndarray,
    distance: float,
    use_global: bool = False,
    inplace: bool = True,
) -> Frame:
    r"""
    Translate a frame along a specified axis by a given distance.

    The origin of the frame is moved along the specified axis by the given distance.

    If ``use_global`` is set to True, the translation is applied in the global coordinate system.
    Otherwise, the translation is applied in the local coordinate system of the frame (in the parent frame).

    .. math::

        \text{new\_origin} = \text{origin} + d \cdot \hat{a}

    where :math:`d` is the distance to translate, and :math:`\hat{a}` is the normalized axis vector.

    .. seealso:: 

        - :class:`py3dframe.Frame` : for more information about the Frame class.
        - :func:`py3dframe.manipulations.rotate_around_axis` : to rotate a frame around a specific axis.
        - :func:`py3dframe.manipulations.translate` : to translate a frame by given offsets along each axis.

    Parameters
    ----------
    frame : Frame
        The frame to translate.

    axis : numpy.ndarray
        A 3D vector representing the axis along which to translate the frame. The vector not need to be normalized.

    distance : float
        The distance to translate along the specified axis. 

    use_global : bool, optional
        If True, the translation is applied in the global coordinate system.
        If False, the translation is applied in the local coordinate system of the frame.
        Default is False.

    inplace : bool, optional
        If True, the translation is applied to the input frame and the same frame is returned.
        If False, a new translated frame is returned and the input frame remains unchanged. Default is True.
    
    Returns
    -------
    Frame
        The translated frame.

        
    Examples
    --------

    .. code-block:: python

        from py3dframe import Frame
        from py3dframe.manipulations import translate_along_axis
        import numpy as np

        # Create a default frame
        frame = Frame.canonical()

        # Define an axis to translate along (e.g., the y-axis)
        axis = np.array([0.0, 1.0, 0.0])

        # Translate the frame by 5 units along the specified axis in the local coordinate system
        translated_frame = translate_along_axis(frame, axis, distance=5.0, use_global=False, inplace=True)

        # The origin of the translated frame is now at (0.0, 5.0, 0.0) in the local coordinate system
        print(translated_frame.origin)  # Output: [0. 5. 0.]

    """
    if not isinstance(frame, Frame):
        raise TypeError("The 'frame' parameter must be an instance of Frame.")
    if not isinstance(distance, Real):
        raise TypeError("The 'distance' parameter must be a real number.")
    if not numpy.isfinite(distance):
        raise ValueError("The 'distance' parameter must be finite.")
    
    axis = numpy.asarray(axis, dtype=float).flatten()
    if axis.shape != (3,):
        raise ValueError("The 'axis' parameter must be a 3-dimensional vector.")
    if not numpy.isfinite(axis).all():
        raise ValueError("The 'axis' parameter must contain finite values.")
    
    norm = numpy.linalg.norm(axis)
    if numpy.abs(norm) < 1e-12:
        raise ValueError("The 'axis' parameter must be a non-zero vector.")
    
    axis_normalized = axis / norm
    translation_vector = (distance * axis_normalized).reshape((3, 1))

    if not inplace:
        frame = frame.copy()

    if use_global:
        frame.global_origin = frame.global_origin + translation_vector
    else:
        frame.origin = frame.origin + translation_vector
    
    return frame