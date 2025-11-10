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

def translate(
    frame: Frame,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    use_global: bool = False,
    inplace: bool = True,
) -> Frame:
    r"""
    Translate a frame by a given offset along each axis.

    The origin of the frame is moved by the specified offsets along the x, y, and z axes.

    If ``use_global`` is set to True, the translation is applied in the global coordinate system.
    Otherwise, the translation is applied in the local coordinate system of the frame (in the parent frame).

    .. math::

        \text{new_origin} = \text{origin} + \begin{bmatrix} dx \\ dy \\ dz \end{bmatrix}

    .. seealso:: 

        - :class:`py3dframe.Frame` : for more information about the Frame class.
        - :func:`py3dframe.manipulations.rotate_around_axis` : to rotate a frame around a specific axis.
        - :func:`py3dframe.manipulations.translate_along_axis` : to translate a frame along a specific axis.

    Parameters
    ----------
    frame : Frame
        The frame to translate.

    dx : float, optional
        The translation along the x-axis. Default is 0.0.

    dy : float, optional
        The translation along the y-axis. Default is 0.0.

    dz : float, optional
        The translation along the z-axis. Default is 0.0.

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
        from py3dframe.manipulations import translate

        # Create a default frame
        frame = Frame.canonical()

        # Translate the frame by 1 unit along the x-axis, 2 units along the y-axis, and 3 units along the z-axis
        translated_frame = translate(frame, dx=1.0, dy=2.0, dz=3.0, use_global=False, inplace=True)

        # The origin of the translated frame is now at (1.0, 2.0, 3.0) in the local coordinate system
        print(translated_frame.origin)  # Output: [1. 2. 3.]

    """
    if not isinstance(frame, Frame):
        raise TypeError("The 'frame' parameter must be an instance of Frame.")
    for value, name in zip((dx, dy, dz), ('dx', 'dy', 'dz')):
        if not isinstance(value, Real):
            raise TypeError(f"The '{name}' parameter must be a real number.")
        if not numpy.isfinite(value):
            raise ValueError(f"The '{name}' parameter must be finite.")
    
    translation_vector = numpy.array([dx, dy, dz]).reshape((3, 1))

    if not inplace:
        frame = frame.copy()

    if use_global:
        frame.global_origin = frame.global_origin + translation_vector
    else:
        frame.origin = frame.origin + translation_vector

    return frame

