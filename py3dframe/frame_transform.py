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

from __future__ import annotations

import numpy 
import scipy
from .frame import Frame
from typing import Optional

from .switch_RT_convention import switch_RT_convention

class FrameTransform:
    r"""
    Class to represent a transformation between two frames of reference.

    Lets consider two orthonormal reference frames :math:`E` and :math:`F` of :math:`\mathbb{R}^3` (see :class:`py3dframe.Frame`).
    The transformation from the frame E (input frame) to the frame F (output frame) can be stored in a FrameTransform object.

    Lets consider a point :math:`X` whose coordinates in the frame E are :math:`\mathbf{X}_i` and in the frame F are :math:`\mathbf{X}_o`.
    There exist 8 principal conventions to express the transformation between the frame E and the frame F.

    The 8 conventions are summarized as follows:

    +---------------------+----------------------------------------------------------------+
    | Index               | Formula                                                        |
    +=====================+================================================================+
    | 0                   | :math:`\mathbf{X}_E = \mathbf{R} \mathbf{X}_F + \mathbf{T}`    |
    +---------------------+----------------------------------------------------------------+
    | 1                   | :math:`\mathbf{X}_E = \mathbf{R} \mathbf{X}_F - \mathbf{T}`    |
    +---------------------+----------------------------------------------------------------+
    | 2                   | :math:`\mathbf{X}_E = \mathbf{R} (\mathbf{X}_F + \mathbf{T})`  |
    +---------------------+----------------------------------------------------------------+
    | 3                   | :math:`\mathbf{X}_E = \mathbf{R} (\mathbf{X}_F - \mathbf{T})`  |
    +---------------------+----------------------------------------------------------------+
    | 4                   | :math:`\mathbf{X}_F = \mathbf{R} \mathbf{X}_E + \mathbf{T}`    |
    +---------------------+----------------------------------------------------------------+
    | 5                   | :math:`\mathbf{X}_F = \mathbf{R} \mathbf{X}_E - \mathbf{T}`    |
    +---------------------+----------------------------------------------------------------+
    | 6                   | :math:`\mathbf{X}_F = \mathbf{R} (\mathbf{X}_E + \mathbf{T})`  |
    +---------------------+----------------------------------------------------------------+
    | 7                   | :math:`\mathbf{X}_F = \mathbf{R} (\mathbf{X}_E - \mathbf{T})`  |
    +---------------------+----------------------------------------------------------------+

    Because the frames are orthonormal, the matrix :math:`\mathbf{R}` is an orthogonal matrix, i.e. :math:`\mathbf{R}^T = \mathbf{R}^{-1}`.

    Parameters
    ----------
    input_frame : Optional[Frame], optional
        The input frame of the transformation. Default is None - the global frame.
    
    output_frame : Optional[Frame], optional
        The output frame of the transformation. Default is None - the global frame.
    
    dynamic : bool, optional
        If True, the transformation will be affected by the changes in the input frame or the output frame. Default is True.
    
    convention : int, optional
        Integer in ``[0, 7]`` selecting the convention to express the transformation. Default is 0.

    Raises  
    ------
    TypeError
        If the input_frame or the output_frame is not a Frame object (or None).
        If the dynamic is not a boolean.
        If the convention is not an integer or a string.
    ValueError
        If the convention is not between 0 and 7.
    
    """
    __slots__ = [
        "_input_frame",
        "_output_frame",
        "_active_input_frame",
        "_active_output_frame",
        "_dynamic",
        "_convention",
    ]

    def __init__(
            self,
            *,
            input_frame: Optional[Frame] = None,
            output_frame: Optional[Frame] = None,
            dynamic: bool = True,
            convention: int = 0,
        ):
        if input_frame is None:
            input_frame = Frame.canonical()
        if output_frame is None:
            output_frame = Frame.canonical()

        self.input_frame = input_frame
        self.output_frame = output_frame
        self.convention = convention
        self.dynamic = dynamic

    
    # ====================================================================================================================================
    # Developer methods
    # ====================================================================================================================================

    @property
    def _R_dev(self) -> scipy.spatial.transform.Rotation:
        """
        Setter for the rotation between the input frame and the output frame in the convention 0.

        The rotation is a scipy.spatial.transform.Rotation object. 

        Returns
        -------
        scipy.spatial.transform.Rotation
            The rotation between the parent frame and the frame in the convention 0.
        """
        # ====================================================================================================================================
        # Lets note : 
        # Xg : the coordinates of a point in the global frame
        # X1 : the coordinates of the same point in the frame 1
        # X2 : the coordinates of the same point in the frame 2
        # R1 : the rotation matrix from the global frame to the frame 1
        # R2 : the rotation matrix from the global frame to the frame 2
        # T1 : the translation vector from the global frame to the frame 1
        # T2 : the translation vector from the global frame to the frame 2
        # R : the rotation matrix from the frame 1 to the frame 2
        # T : the translation vector from the frame 1 to the frame 2
        # 
        # We have :
        # Xg = R1 * X1 + T1
        # Xg = R2 * X2 + T2
        # X1 = R * X2 + T
        #
        # We search R:
        # X1 = R1.inv() * (Xg - T1)
        # X1 = R1.inv() * (R2 * X2 + T2 - T1)
        # X1 = R1.inv() * R2 * X2 + R1.inv() * (T2 - T1)
        # So R = R1.inv() * R2
        # ====================================================================================================================================
        R_input = self._get_active_input_frame().get_global_rotation(convention=0)
        R_output = self._get_active_output_frame().get_global_rotation(convention=0)
        return R_input.inv() * R_output
    
    @property
    def _T_dev(self) -> numpy.ndarray:
        """
        Getter and setter for the translation vector between the parent frame and the frame in the convention 0.

        The translation vector is a 3-element vector.

        .. warning::

            The T_dev attribute is flags.writeable = False. To change the translation vector, use the setter.

        Returns
        -------
        numpy.ndarray
            The translation vector between the parent frame and the frame in the convention 0 with shape (3, 1).
        """
        # ====================================================================================================================================
        # Lets note : 
        # Xg : the coordinates of a point in the global frame
        # X1 : the coordinates of the same point in the frame 1
        # X2 : the coordinates of the same point in the frame 2
        # R1 : the rotation matrix from the global frame to the frame 1
        # R2 : the rotation matrix from the global frame to the frame 2
        # T1 : the translation vector from the global frame to the frame 1
        # T2 : the translation vector from the global frame to the frame 2
        # R : the rotation matrix from the frame 1 to the frame 2
        # T : the translation vector from the frame 1 to the frame 2
        # 
        # We have :
        # Xg = R1 * X1 + T1
        # Xg = R2 * X2 + T2
        # X1 = R * X2 + T
        #
        # We search T:
        # X1 = R1.inv() * (Xg - T1)
        # X1 = R1.inv() * (R2 * X2 + T2 - T1)
        # X1 = R1.inv() * R2 * X2 + R1.inv() * (T2 - T1)
        # So T = R1.inv() * (T2 - T1)
        # ====================================================================================================================================
        R_input = self._get_active_input_frame().get_global_rotation(convention=0)
        T_input = self._get_active_input_frame().get_global_translation(convention=0)
        T_output = self._get_active_output_frame().get_global_translation(convention=0)
        return R_input.inv().apply((T_output - T_input).T).T


    def _set_active_input_frame(self, frame: Frame) -> None:
        r"""
        Set the active input frame of the transformation.

        If the ``dynamic`` attribute is set to ``True``, the new active input frame is set to the given frame.

        If the ``dynamic`` attribute is set to ``False``, the method ignores the given frame and the active input frame stay the same.

        Parameters
        ----------
        frame : Frame
            The new active input frame of the transformation.
        """
        if not isinstance(frame, Frame):
            raise TypeError("The frame must be a Frame object.")
        self._active_input_frame = frame.get_global_frame() # copy the frame to avoid modification of the active frame when the user modifies the input frame

    def _get_active_input_frame(self) -> Frame:
        r"""
        Get the active input frame of the transformation.

        If the ``dynamic`` attribute is set to ``True``, the active input frame is the current input frame.

        If the ``dynamic`` attribute is set to ``False``, the active input frame is the input frame at the time of the creation of the transformation or at the time of the change of the ``dynamic`` attribute to ``False``.

        Returns
        -------
        Frame
            The active input frame of the transformation.
        """
        if self.dynamic:
            return self.input_frame
        return self._active_input_frame
    
    def _set_active_output_frame(self, frame: Frame) -> None:
        r"""
        Set the active output frame of the transformation.

        If the ``dynamic`` attribute is set to ``True``, the new active output frame is set to the given frame.

        If the ``dynamic`` attribute is set to ``False``, the method ignores the given frame and the active output frame stay the same.

        Parameters
        ----------
        frame : Frame
            The new active output frame of the transformation.
        """
        if not isinstance(frame, Frame):
            raise TypeError("The frame must be a Frame object.")
        self._active_output_frame = frame.get_global_frame() # copy the frame to avoid modification of the active frame when the user modifies the output frame

    def _get_active_output_frame(self) -> Frame:
        r"""
        Get the active output frame of the transformation.

        If the ``dynamic`` attribute is set to ``True``, the active output frame is the current output frame.

        If the ``dynamic`` attribute is set to ``False``, the active output frame is the output frame at the time of the creation of the transformation or at the time of the change of the ``dynamic`` attribute to ``False``.

        Returns
        -------
        Frame
            The active output frame of the transformation.
        """
        if self.dynamic:
            return self.output_frame
        return self._active_output_frame



    # ====================================================================================================================================
    # Public methods
    # ====================================================================================================================================

    @property
    def input_frame(self) -> Frame:
        r"""
        The input frame of the transformation.

        .. note::

            This attribute is settable.

        The ``input_frame`` attribute is a Frame object.

        .. warning::

            ``dynamic`` attribute must be set to True to take into account the changes in the input frame.
        
        .. seealso::

            - :attr:`output_frame` for the output frame of the transformation.
            - :attr:`dynamic` for the dynamic attribute of the transformation.
            - :meth:`get_active_input_frame` to extract the active input frame of the transformation if ``dynamic`` is set to ``True``.

        Parameters
        ----------
        input_frame : Optional[Frame]
            The input frame of the transformation. If None, the global canonical frame is used.

        Returns
        -------
        Frame
            The input frame of the transformation.

        Examples
        --------
        Lets create a FrameTransform object with the global frame as input frame and a local frame as output frame.

        .. code-block:: python

            from py3dframe import Frame, FrameTransform

            frame_E = Frame.canonical() # Input frame - Global frame
            frame_F = Frame.from_axes(origin=[1, 2, 3], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1]) # Output frame - Local frame

            transform = FrameTransform(input_frame=frame_E, output_frame=frame_F, dynamic=True, convention=0)

        To extract the input frame of the transformation, use the ``input_frame`` property.

        .. code-block:: python

            input_frame = transform.input_frame
            print(input_frame)
            # Output: Frame(origin=[[0.] [0.] [0.]], x_axis=[[1.] [0.] [0.]], y_axis=[[0.] [1.] [0.]], z_axis=[[0.] [0.] [1.]])

        The user can also change the input frame of the transformation.

        .. code-block:: python

            new_input_frame = Frame.from_axes(origin=[[1.] [1.] [1.]], x_axis=[[0.] [1.] [0.]], y_axis=[[0.] [0.] [1.]], z_axis=[[1.] [0.] [0.]])
            transform.input_frame = new_input_frame

        Now the transformation allows to transform points from the new input frame to the output frame.

        """
        return self._input_frame
    
    @input_frame.setter
    def input_frame(self, input_frame: Optional[Frame]) -> None:
        if input_frame is None:
            input_frame = Frame.canonical()
        if not isinstance(input_frame, Frame):
            raise TypeError("The input_frame must be a Frame object.")
        self._input_frame = input_frame

    @property
    def output_frame(self) -> Frame:
        r"""
        The output frame of the transformation.

        .. note::

            This attribute is settable.

        The ``output_frame`` attribute is a Frame object.

        .. warning::

            ``dynamic`` attribute must be set to True to take into account the changes in the output frame.

        .. seealso::

            - :attr:`input_frame` for the input frame of the transformation.
            - :attr:`dynamic` for the dynamic attribute of the transformation.
            - :meth:`get_active_output_frame` to extract the active output frame of the transformation if ``dynamic`` is set to ``True``.

        Parameters
        ----------
        output_frame : Optional[Frame]
            The output frame of the transformation. If None, the global canonical frame is used.

        Returns
        -------
        Frame
            The output frame of the transformation.

        Examples
        --------
        Lets create a FrameTransform object with the global frame as input frame and a local frame as output frame.

        .. code-block:: python

            from py3dframe import Frame, FrameTransform

            frame_E = Frame.canonical() # Input frame - Global frame
            frame_F = Frame.from_axes(origin=[1, 2, 3], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1]) # Output frame - Local frame

            transform = FrameTransform(input_frame=frame_E, output_frame=frame_F, dynamic=True, convention=0)

        To extract the output frame of the transformation, use the ``output_frame`` property.

        .. code-block:: python

            output_frame = transform.output_frame
            print(output_frame)
            # Output: Frame(origin=[[1.] [2.] [3.]], x_axis=[[1.] [0.] [0.]], y_axis=[[0.] [1.] [0.]], z_axis=[[0.] [0.] [1.]])

        The user can also change the output frame of the transformation.

        .. code-block:: python

            new_output_frame = Frame.from_axes(origin=[[2.] [2.] [2.]], x_axis=[[0.] [1.] [0.]], y_axis=[[0.] [0.] [1.]], z_axis=[[1.] [0.] [0.]])
            transform.output_frame = new_output_frame
        
        Now the transformation allows to transform points from the input frame to the new output frame.

        """
        return self._output_frame
    
    @output_frame.setter
    def output_frame(self, output_frame: Optional[Frame]) -> None:
        if output_frame is None:
            output_frame = Frame.canonical()
        if not isinstance(output_frame, Frame):
            raise TypeError("The output_frame must be a Frame object.")
        self._output_frame = output_frame

    @property
    def dynamic(self) -> bool:
        r"""
        The dynamic attribute controls whether the transformation is affected by changes in the input or output frame.

        .. note::

            This attribute is settable.

        The dynamic attribute is a boolean. If True, the transformation will be affected by the changes in the input frame or the output frame.
        Otherwise, the transformation will be frozen at the time of the transformation object creation or at the time of the change of the dynamic attribute to False.

        Parameters
        ----------
        dynamic : bool
            If True, the transformation will be affected by the changes in the input frame or the output frame.

        Returns
        -------
        bool
            If True, the transformation will be affected by the changes in the input frame or the output frame.

        Examples
        --------
        Lets create a FrameTransform object with the global frame as input frame and a local frame as output frame.

        .. code-block:: python

            import numpy as np
            from py3dframe import Frame, FrameTransform
            frame_E = Frame.canonical() # Input frame - Global frame
            frame_F = Frame.from_axes(origin=[1, 2, 3], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1]) # Output frame - Local frame

            transform = FrameTransform(input_frame=frame_E, output_frame=frame_F, dynamic=True, convention=0)

        Here ``dynamic`` is set to ``True``, so the transformation will be affected by the changes in the input frame or the output frame.

        .. code-block:: python

            X_i = np.array([1, 2, 3]).reshape((3, 1))
            X_o = transform.transform(point=X_i)
            print(X_o)
            # Output: [[0.] [0.] [0.]]

            print(transform.input_frame)
            # Output: Frame(origin=[[0.] [0.] [0.]], x_axis=[[1.] [0.] [0.]], y_axis=[[0.] [1.] [0.]], z_axis=[[0.] [0.] [1.]])
            print(transform.get_active_input_frame())
            # Output: Frame(origin=[[0.] [0.] [0.]], x_axis=[[1.] [0.] [0.]], y_axis=[[0.] [1.] [0.]], z_axis=[[0.] [0.] [1.]])

            # Now, we change the input frame
            frame_E.origin = [1, 1, 1]

            X_o = transform.transform(point=X_i)
            print(X_o)
            # Output: [[1.] [1.] [1.]]

            print(transform.input_frame)
            # Output: Frame(origin=[[1.] [1.] [1.]], x_axis=[[1.] [0.] [0.]], y_axis=[[0.] [1.] [0.]], z_axis=[[0.] [0.] [1.]])
            print(transform.get_active_input_frame())
            # Output: Frame(origin=[[1.] [1.] [1.]], x_axis=[[1.] [0.] [0.]], y_axis=[[0.] [1.] [0.]], z_axis=[[0.] [0.] [1.]])

        Lets set ``dynamic`` to ``False``, any modification of the input frame or the output frame will not affect the transformation.

        .. code-block:: python

            transform.dynamic = False

            X_i = [1, 2, 3]
            X_o = transform.transform(point=X_i)
            print(X_o)
            # Output: [[1.] [1.] [1.]]

            print(transform.input_frame)
            # Output: Frame(origin=[[1.] [1.] [1.]], x_axis=[[1.] [0.] [0.]], y_axis=[[0.] [1.] [0.]], z_axis=[[0.] [0.] [1.]])
            print(transform.get_active_input_frame())
            # Output: Frame(origin=[[1.] [1.] [1.]], x_axis=[[1.] [0.] [0.]], y_axis=[[0.] [1.] [0.]], z_axis=[[0.] [0.] [1.]])

            # Now, we change the input frame
            frame_E.origin = [2, 2, 2]

            X_o = transform.transform(point=X_i)
            print(X_o)
            # Output: [[1.] [1.] [1.]]

            print(transform.input_frame)
            # Output: Frame(origin=[[2.] [2.] [2.]], x_axis=[[1.] [0.] [0.]], y_axis=[[0.] [1.] [0.]], z_axis=[[0.] [0.] [1.]])
            print(transform.get_active_input_frame())
            # Output: Frame(origin=[[1.] [1.] [1.]], x_axis=[[1.] [0.] [0.]], y_axis=[[0.] [1.] [0.]], z_axis=[[0.] [0.] [1.]])

        .. warning::

            Notice that when ``dynamic`` is set to ``True``, the transformation is unchanged when the user gives a new input frame or a new output frame. 
            However, the input frame or the output frame of the transformation is updated.
            Use the :meth:`get_active_input_frame` or :meth:`get_active_output_frame` methods to extract the active input frame or the active output frame of the transformation.

        When ``dynamic`` is set to ``False``, if the user gives a new input frame or a new output frame, the transformation will unchanged too.

        .. code-block:: python

            new_input_frame = Frame.from_axes(origin=[3, 3, 3], x_axis=[0, 1, 0], y_axis=[0, 0, 1], z_axis=[1, 0, 0])
            transform.input_frame = new_input_frame
            print(transform.input_frame)
            # Output: Frame(origin=[[3.] [3.] [3.]], x_axis=[[0.] [1.] [0.]], y_axis=[[0.] [0.] [1.]], z_axis=[[1.] [0.] [0.]])
            print(transform.get_active_input_frame())
            # Output: Frame(origin=[[1.] [1.] [1.]], x_axis=[[1.] [0.] [0.]], y_axis=[[0.] [1.] [0.]], z_axis=[[0.] [0.] [1.]])

        When ``dynamic`` is set to to ``True``, the transformation instantly takes into account the new input frame or the new output frame.

        .. code-block:: python

            transform.dynamic = True
            print(transform.input_frame)
            # Output: Frame(origin=[[3.] [3.] [3.]], x_axis=[[0.] [1.] [0.]], y_axis=[[0.] [0.] [1.]], z_axis=[[1.] [0.] [0.]])
            print(transform.get_active_input_frame())
            # Output: Frame(origin=[[3.] [3.] [3.]], x_axis=[[0.] [1.] [0.]], y_axis=[[0.] [0.] [1.]], z_axis=[[1.] [0.] [0.]])

        """
        return self._dynamic

    @dynamic.setter
    def dynamic(self, dynamic: bool) -> None:
        if not isinstance(dynamic, bool):
            raise TypeError("The dynamic must be a boolean.")
        if not dynamic:
            self._set_active_input_frame(self._input_frame)
            self._set_active_output_frame(self._output_frame)
        if dynamic:
            self._active_input_frame = None
            self._active_output_frame = None
        self._dynamic = dynamic


    @property
    def convention(self) -> int:
        r"""
        The convention to express the transformation between the input frame and the output frame.

        .. note::

            This property is settable.

        The convention can be an integer between 0 and 7 corresponding to the conventions described in the table below:

        +---------------------+----------------------------------------------------------------+
        | Index               | Formula                                                        |
        +=====================+================================================================+
        | 0                   | :math:`\mathbf{X}_E = \mathbf{R} \mathbf{X}_F + \mathbf{T}`    |
        +---------------------+----------------------------------------------------------------+
        | 1                   | :math:`\mathbf{X}_E = \mathbf{R} \mathbf{X}_F - \mathbf{T}`    |
        +---------------------+----------------------------------------------------------------+
        | 2                   | :math:`\mathbf{X}_E = \mathbf{R} (\mathbf{X}_F + \mathbf{T})`  |
        +---------------------+----------------------------------------------------------------+
        | 3                   | :math:`\mathbf{X}_E = \mathbf{R} (\mathbf{X}_F - \mathbf{T})`  |
        +---------------------+----------------------------------------------------------------+
        | 4                   | :math:`\mathbf{X}_F = \mathbf{R} \mathbf{X}_E + \mathbf{T}`    |
        +---------------------+----------------------------------------------------------------+
        | 5                   | :math:`\mathbf{X}_F = \mathbf{R} \mathbf{X}_E - \mathbf{T}`    |
        +---------------------+----------------------------------------------------------------+
        | 6                   | :math:`\mathbf{X}_F = \mathbf{R} (\mathbf{X}_E + \mathbf{T})`  |
        +---------------------+----------------------------------------------------------------+
        | 7                   | :math:`\mathbf{X}_F = \mathbf{R} (\mathbf{X}_E - \mathbf{T})`  |
        +---------------------+----------------------------------------------------------------+

        Where:

        - :math:`\mathbf{X}_E` is a point expressed in the input frame coordinates.
        - :math:`\mathbf{X}_F` is the same point expressed in the output frame coordinates.
        - :math:`\mathbf{R}` is the rotation matrix between the input frame and the output frame.
        - :math:`\mathbf{T}` is the translation vector between the input frame and the output frame.

        .. note::

            The default convention is 0.

        Parameters
        ----------
        convention : int
            Integer in ``[0, 7]`` selecting the convention to express the transformation between the input frame and the output frame.

        Returns
        -------
        int
            The convention parameter.
        """
        return self._convention
    
    @convention.setter
    def convention(self, convention: int) -> None:
        if not isinstance(convention, int):
            raise TypeError("The convention parameter must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        self._convention = convention

    
    def get_active_input_frame(self) -> Frame:
        r"""
        Get a copy of the active input frame of the transformation.

        If the ``dynamic`` attribute is set to ``True``, the active input frame is the current input frame.

        If the ``dynamic`` attribute is set to ``False``, the active input frame is the input frame at the time of the creation of the transformation or at the time of the change of the ``dynamic`` attribute to ``False``.

        .. note::

            A copy of the active input frame is returned to avoid modification of the active input frame when the user modifies the returned frame.

        .. seealso::

            - :attr:`input_frame` for the input frame of the transformation.
            - :attr:`dynamic` for the dynamic attribute of the transformation.

        Returns
        -------
        Frame
            The active input frame of the transformation.
        """
        return self._get_active_input_frame().copy()
    
    def get_active_output_frame(self) -> Frame:
        r"""
        Get a copy of the active output frame of the transformation.

        If the ``dynamic`` attribute is set to ``True``, the active output frame is the current output frame.

        If the ``dynamic`` attribute is set to ``False``, the active output frame is the output frame at the time of the creation of the transformation or at the time of the change of the ``dynamic`` attribute to ``False``.

        .. note::

            A copy of the active output frame is returned to avoid modification of the active output frame when the user modifies the returned frame.

        .. seealso::

            - :attr:`output_frame` for the output frame of the transformation.
            - :attr:`dynamic` for the dynamic attribute of the transformation.

        Returns
        -------
        Frame
            The active output frame of the transformation.
        """
        return self._get_active_output_frame().copy()

    
    def get_rotation(self, *, convention: Optional[int] = None) -> scipy.spatial.transform.Rotation:
        r"""
        Get the rotation between the input frame and the output frame in a specified convention.

        Parameters
        ----------
        convention : Optional[int], optional
           Integer in ``[0, 7]`` selecting the convention to express the transformation between the input frame and the output frame. Default is the convention of the transformation.

        Returns
        -------
        Rotation
            The rotation between the input frame and the output frame in the specified convention.
        """
        if convention is None:
            convention = self._convention
        if not isinstance(convention, int):
            raise TypeError("The convention must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        R, _ = switch_RT_convention(self._R_dev, self._T_dev, 0, convention)
        return R

    @property
    def rotation(self) -> scipy.spatial.transform.Rotation:
        r"""
        Getter for the rotation between the input frame and the output frame in the convention of the transformation.

        .. seealso::

            - method :meth:`py3dframe.FrameTransform.get_rotation` to get the rotation in a specific convention.

        Returns
        -------
        Rotation
            The rotation between the input frame and the output frame in the convention of the transformation.
        """
        return self.get_rotation(convention=self._convention)

    def get_translation(self, *, convention: Optional[int] = None) -> numpy.ndarray:
        r"""
        Get the translation vector between the input frame and the output frame in a specified convention.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention to express the transformation between the input frame and the output frame. Default is the convention of the transformation.

        Returns
        -------
        numpy.ndarray
            The translation vector between the input frame and the output frame in the specified convention with shape (3, 1).
        """
        if convention is None:
            convention = self._convention
        if not isinstance(convention, int):
            raise TypeError("The convention must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        _, T = switch_RT_convention(self._R_dev, self._T_dev, 0, convention)
        return T

    @property
    def translation(self) -> numpy.ndarray:
        r"""
        Getter for the translation vector between the input frame and the output frame in the convention of the transformation.

        .. seealso::

            - method :meth:`py3dframe.FrameTransform.get_translation` to get the translation vector in a specific convention.

        Returns
        -------
        numpy.ndarray
            The translation vector between the input frame and the output frame in the convention of the transformation with shape (3, 1).
        """
        return self.get_translation(convention=self._convention)

    def get_rotation_matrix(self, *, convention: Optional[int] = None) -> numpy.ndarray:
        r"""
        Get the rotation matrix representation of the rotation between the input frame and the output frame in a specified convention.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention to express the transformation between the input frame and the output frame. Default is the convention of the transformation.

        Returns
        -------
        numpy.ndarray
            The rotation matrix between the input frame and the output frame in the specified convention with shape (3, 3).
        """
        return self.get_rotation(convention=convention).as_matrix()

    @property
    def rotation_matrix(self) -> numpy.ndarray:
        r"""
        Getter for the rotation matrix representation of the rotation between the input frame and the output frame in the convention of the transformation.

        .. seealso::

            - method :meth:`py3dframe.FrameTransform.get_rotation_matrix` to get the rotation matrix in a specific convention.

        Returns
        -------
        numpy.ndarray
            The rotation matrix between the input frame and the output frame in the convention of the transformation with shape (3, 3).
        """
        return self.get_rotation_matrix(convention=self._convention)

    def get_quaternion(self, *, convention: Optional[int] = None, scalar_first: bool = True) -> numpy.ndarray:
        r"""
        Get the quaternion representation of the rotation between the input frame and the output frame in a specified convention.
        
        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention to express the transformation between the input frame and the output frame. Default is the convention of the transformation.

        scalar_first : bool, optional
            If True, the quaternion will be in the scalar-first convention. Default is True.

        Returns
        -------
        numpy.ndarray
            The quaternion between the input frame and the output frame in the specified convention with shape (4,).
        """
        if not isinstance(scalar_first, bool):
            raise TypeError("The scalar_first must be a boolean.")
        return self.get_rotation(convention=convention).as_quat(scalar_first=scalar_first)
    
    @property
    def quaternion(self) -> numpy.ndarray:
        r"""
        Getter for the quaternion representation of the rotation between the input frame and the output frame in the convention of the transformation.

        The quaternion is in the scalar-first convention.

        .. seealso::

            - method :meth:`py3dframe.FrameTransform.get_quaternion` to get the quaternion in a specific convention.

        Returns
        -------
        numpy.ndarray
            The quaternion between the input frame and the output frame in the convention of the transformation with shape (4,) in the scalar-first convention.

        """
        return self.get_quaternion(convention=self._convention, scalar_first=True)


    def get_euler_angles(self, *, convention: Optional[int] = None, seq: str = 'xyz', degrees: bool = False) -> numpy.ndarray:
        r"""
        Get the Euler angles representation of the rotation between the input frame and the output frame in a specified convention.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention to express the transformation between the input frame and the output frame. Default is the convention of the transformation.

        seq : str, optional
            The sequence of axes for the Euler angles. It must be a string of length 3 containing only the characters 'X', 'Y', 'Z', 'x', 'y', 'z'. Default is 'xyz'.

        degrees : bool, optional
            If True, the Euler angles will be in degrees. Default is False (radians).

        Returns
        -------
        numpy.ndarray
            The Euler angles between the input frame and the output frame in the specified convention with shape (3,).
        """
        if not isinstance(seq, str):
            raise TypeError("The seq must be a string.")
        if not isinstance(degrees, bool):
            raise TypeError("The degrees must be a boolean.")
        if not len(seq) == 3:
            raise ValueError("The seq must be a string of length 3.")
        if not all([s in 'XYZxyz' for s in seq]):
            raise ValueError("The seq must contain only the characters 'X', 'Y', 'Z', 'x', 'y', 'z'.") 
        return self.get_rotation(convention=convention).as_euler(seq, degrees=degrees)
    
    @property
    def euler_angles(self) -> numpy.ndarray:
        r"""
        Getter for the Euler angles representation of the rotation between the input frame and the output frame in the convention of the transformation.

        The Euler angles are in radians and in the 'xyz' sequence.

        .. seealso::

            - method :meth:`py3dframe.FrameTransform.get_euler_angles` to get the Euler angles in a specific convention, sequence or unit.

        Returns
        -------
        numpy.ndarray
            The Euler angles between the input frame and the output frame in the convention of the transformation with shape (3,) in radians and in the 'xyz' sequence.
        """
        return self.get_euler_angles(convention=self._convention, seq='xyz', degrees=False)
    


    def get_rotation_vector(self, *, convention: Optional[int] = None, degrees: bool = False) -> numpy.ndarray:
        r"""
        Get the rotation vector representation of the rotation between the input frame and the output frame in a specified convention.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention to express the transformation between the input frame and the output frame. Default is the convention of the transformation.

        degrees : bool, optional
            If True, the rotation vector will be in degrees. Default is False (radians).

        Returns
        -------
        numpy.ndarray
            The rotation vector between the input frame and the output frame in the specified convention with shape (3,).
        """
        if not isinstance(degrees, bool):
            raise TypeError("The degrees must be a boolean.")
        return self.get_rotation(convention=convention).as_rotvec(degrees=degrees)

    @property
    def rotation_vector(self) -> numpy.ndarray:
        r"""
        Getter for the rotation vector representation of the rotation between the input frame and the output frame in the convention of the transformation.

        The rotation vector is in radians.

        .. seealso::

            - method :meth:`py3dframe.FrameTransform.get_rotation_vector` to get the rotation vector in a specific convention or unit.

        Returns
        -------
        numpy.ndarray
            The rotation vector between the input frame and the output frame in the convention of the transformation with shape (3,) in radians.
        """
        return self.get_rotation_vector(convention=self._convention, degrees=False)

    
    def transform(self, *, point: Optional[numpy.ndarray] = None, vector: Optional[numpy.ndarray] = None) -> numpy.ndarray:
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

        Parameters
        ----------
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
    


    def inverse_transform(self, *, point: Optional[numpy.ndarray] = None, vector: Optional[numpy.ndarray] = None) -> numpy.ndarray:
        r"""
        Transform a point or a vector from the output frame to the input frame.

        If the point is provided, the method will return the coordinates of the point in the input frame.
        If the vector is provided, the method will return the coordinates of the vector in the input frame.

        Several points / vectors can be transformed at the same time by providing a 2D numpy array with shape (3, N).

        If both the point and the vector are provided, the method will raise a ValueError.
        If neither the point nor the vector is provided, the method will return None.

        In the convention 0:

        .. math::

            X_{\text{input_frame}} = R * X_{\text{output_frame}} + T

        .. math::

            V_{\text{input_frame}} = R * V_{\text{output_frame}}

        Parameters
        ----------
        point : Optional[array_like], optional
            The coordinates of the point in the output frame with shape (3, N). Default is None.
        
        vector : Optional[array_like], optional
            The coordinates of the vector in the output frame with shape (3, N). Default is None.

        Returns
        -------
        numpy.ndarray
            The coordinates of the point or the vector in the input frame with shape (3, N).

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

        The FrameTransform object can be used to transform points or vectors from the output frame to the input frame.

        .. code-block:: python

            X_o = np.array([0, 0, 0]).reshape((3, 1)) # Point in the output frame coordinates
            X_i = transform.inverse_transform(point=X_o) # Transform the point to the input frame coordinates
            print(X_i)
            # Output: [[1.] [2.] [3.]]

            V_o = np.array([1, 0, 0]).reshape((3, 1)) # Vector in the output frame coordinates
            V_i = transform.inverse_transform(vector=V_o) # Transform the vector to the input frame coordinates
            print(V_i)
            # Output: [[1.] [0.] [0.]]
            
        """
        if point is not None and vector is not None:
            raise ValueError("Only one of 'point' or 'vector' can be provided.")
        if point is None and vector is None:
            return None
        
        output_data = point if point is not None else vector
        output_data = numpy.array(output_data).astype(numpy.float64)

        if not output_data.ndim == 2 or output_data.shape[0] != 3:
            raise ValueError("The points or vectors must be a 2D numpy array with shape (3, N).")

        # Convert the output data to vector input
        input_data = self._R_dev.apply(output_data.T).T

        # Convert the vector to point
        if point is not None:
            input_data = input_data + self._T_dev

        return input_data
    
    
    def inverse(self) -> FrameTransform:
        r"""
        Get the inverse transformation by swapping the input frame and the output frame.

        Returns
        -------
        FrameTransform
            The inverse transformation.
        """
        new_transform = FrameTransform(input_frame=self._output_frame, output_frame=self._input_frame, dynamic=self._dynamic, convention=self._convention)
        if self._dynamic:
            new_transform._set_active_input_frame(self.get_active_output_frame())
            new_transform._set_active_output_frame(self.get_active_input_frame())
        return new_transform


        