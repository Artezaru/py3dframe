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

from .frame import Frame
import numpy

from typing import Optional

def transform(
    input_frame: Optional[Frame],
    output_frame: Optional[Frame],
    *,
    point: Optional[numpy.ndarray] = None,
    vector: Optional[numpy.ndarray] = None,
) -> numpy.ndarray:
    r"""
    Transform a point or a vector from the input frame to the output frame.

    If the point is provided, the method will return the coordinates of the point in the output frame.
    If the vector is provided, the method will return the coordinates of the vector in the output frame.

    Several points / vectors can be transformed at the same time by providing a 2D numpy array with shape (3, N).

    If both the point and the vector are provided, the method will raise a ValueError.
    If neither the point nor the vector is provided, the method will return None.

    In the convention 0:

    .. math::

        X_{\text{output_frame}} = R^{-1} * (X_{\text{input_frame}} - T)

    .. math::

        V_{\text{output_frame}} = R^{-1} * V_{\text{input_frame}}

    .. seealso::

        - :class:`py3dframe.Frame` : for more information about the Frame class.
        - :class:`py3dframe.FrameTransform` : to store and reuse frame transformations.

    Parameters
    ----------
    input_frame : Optional[Frame]
        The input frame. If None, the canonical frame is assumed.

    output_frame : Optional[Frame]
        The output frame. If None, the canonical frame is assumed.

    point : Optional[array_like], optional
        The coordinates of the point in the input frame with shape (3, N). Default is None.
    
    vector : Optional[array_like], optional
        The coordinates of the vector in the input frame with shape (3, N). Default is None.

    Returns
    -------
    numpy.ndarray
        The coordinates of the point or the vector in the output frame with shape (3, N).

    Raises
    ------
    ValueError
        If the point or the vector is not provided.
        If point and vector are both provided.

    Examples
    --------
    Lets create a FrameTransform object with the global frame as input frame and a local frame as output frame.

    .. code-block:: python

        import numpy as np
        from py3dframe import Frame, FrameTransform

        frame_E = Frame.canonical() # Input frame - Global frame
        frame_F = Frame.from_axes(origin=[1, 2, 3], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1]) # Output frame - Local frame

        transform = FrameTransform(input_frame=frame_E, output_frame=frame_F, dynamic=True, convention=0)

    The FrameTransform object can be used to transform points or vectors from the input frame to the output frame.

    .. code-block:: python

        X_i = np.array([1, 2, 3]).reshape((3, 1)) # Point in the input frame coordinates
        X_o = transform.transform(point=X_i) # Transform the point to the output frame coordinates
        print(X_o)
        # Output: [[0.] [0.] [0.]]

        V_i = np.array([1, 0, 0]).reshape((3, 1)) # Vector in the input frame coordinates
        V_o = transform.transform(vector=V_i) # Transform the vector to the output frame coordinates
        print(V_o)
        # Output: [[1.] [0.] [0.]]

    """
    if input_frame is not None and not isinstance(input_frame, Frame):
        raise TypeError("The 'input_frame' parameter must be an instance of Frame or None.")  
    if output_frame is not None and not isinstance(output_frame, Frame):
        raise TypeError("The 'output_frame' parameter must be an instance of Frame or None.")

    if point is not None and vector is not None:
        raise ValueError("Only one of 'point' or 'vector' can be provided.")
    if point is None and vector is None:
        return None
    
    input_data = point if point is not None else vector
    input_data = numpy.array(input_data).astype(numpy.float64)

    if not input_data.ndim == 2 or input_data.shape[0] != 3:
        raise ValueError("The points or vectors must be a 2D numpy array with shape (3, N).")

    # Convert the point to vector
    if point is not None:
        input_data = input_data - self._T_dev
    
    # Convert the input data to the output frame
    output_data = self._R_dev.inv().apply(input_data.T).T

    return output_data