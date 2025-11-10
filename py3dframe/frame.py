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

from typing import Optional, Dict, Any, Union, Sequence

import numpy 
import scipy
import json
import copy

from .switch_RT_convention import switch_RT_convention
from .matrix import is_SO3
from .rotation import Rotation

class Frame:
    r"""
    A Frame object represents a frame of reference of :math:`\mathbb{R}^3`.
    
    .. warning::

        Only right-handed orthogonal frames are supported (orthonormal basis vectors with a positive determinant).

    A frame of reference is defined by 
    
    - An origin :math:`O_F` as a 3-element vector.
    - Three basis vectors :math:`\mathbf{e}_1`, :math:`\mathbf{e}_2` and :math:`\mathbf{e}_3` as 3-element vectors.

    For example:

    .. code-block:: python

        import numpy
        from py3dframe import Frame
        e1 = [1, 0, 0]
        e2 = [0, 1, 0]
        e3 = [0, 0, 1]
        origin = [1, 2, 3]
        frame = Frame.from_axes(origin=origin, x_axis=e1, y_axis=e2, z_axis=e3)

    The frame can also be defined by its transformation relative to its parent frame (by default the global canonical frame of :math:`\mathbb{R}^3`).
    Lets note :math:`E` the parent frame (or the global frame of :math:`\mathbb{R}^3`) and :math:`F` the frame to define.
    The transformation between the frame :math:`E` and the frame :math:`F` is defined by a rotation matrix :math:`R` and a translation vector :math:`T`.
    Several conventions can be used to define the transformation between the frames.

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

    .. note::

        If the axes are provided, the class will normalize the basis vectors to get an orthonormal matrix.    

    .. seealso::

        - class :class:`py3dframe.FrameTransform` to represent a transformation between two frames of reference and convert points between the frames.

    Parameters
    ----------
    translation : array_like, optional
        The translation vector between the parent frame and the frame to create in the selected convention.
        The translation vector is a 3-element vector and the default value is the zero vector.

    rotation: Rotation, optional
        The rotation between the parent frame and the frame to create in the selected convention.
        The rotation is a scipy.spatial.transform.Rotation object and the default value is the identity rotation.

    parent : Optional[Frame], optional
        If given, the frame will be defined relatively to this parent frame. Default is None - the global frame of :math:`\mathbb{R}^3`.

    setup_only : bool, optional
        If True, the parent frame will be used only to define the frame. Once the frame is created, its global position and orientation will be computed and the parent frame will be unlinked (set to None).

    convention : int, optional
        Integer in ``[0, 7]`` selecting the convention to express the transformation. Default is 0.
    
    Raises
    ------
    TypeError
        If any of the parameters is wrong type.

    """
    __slots__ = [
        "__R_dev",
        "__T_dev",
        "_parent",
        "_convention",
    ]

    def __init__(
            self,
            translation: Optional[numpy.ndarray] = None,
            rotation: Optional[scipy.spatial.transform.Rotation] = None,
            *,
            parent: Optional[Frame] = None,
            setup_only: bool = False,
            convention: Optional[int] = 0,
        ) -> None:
        # Initialize a Canonical Frame
        self._R_dev = Rotation.from_matrix(numpy.eye(3))
        self._T_dev = numpy.zeros((3, 1), dtype=numpy.float64)
        self.parent = parent
        self.convention = convention

        # Update the frame with the given parameters
        translation = translation if translation is not None else numpy.zeros((3,1), dtype=numpy.float64)
        rotation = rotation if rotation is not None else Rotation.identity()
        self.set_translation(translation, convention=convention)
        self.set_rotation(rotation, convention=convention)

        if setup_only:
            self._remove_parent_dev()

    # ====================================================================================================================================
    # Developer methods
    # ====================================================================================================================================
    @property
    def _R_dev(self) -> scipy.spatial.transform.Rotation:
        r"""
        Getter and setter for the rotation object between the parent frame and this frame in the convention 0.

        The rotation is a scipy.spatial.transform.Rotation object. 

        Returns
        -------
        scipy.spatial.transform.Rotation
            The rotation between the parent frame and this frame in the convention 0.
        """
        return self.__R_dev
    
    @_R_dev.setter
    def _R_dev(self, R: scipy.spatial.transform.Rotation) -> None:
        if not isinstance(R, scipy.spatial.transform.Rotation):
            raise TypeError("The rotation must be a scipy.spatial.transform.Rotation object.")
        self.__R_dev = R
    

    @property
    def _T_dev(self) -> numpy.ndarray:
        r"""
        Getter and setter for the translation vector between the parent frame and this frame in the convention 0.

        The translation vector is a 3-element vector.

        .. warning::

            The T_dev attribute is flags.writeable = False. To change the translation vector, use the setter.

        Returns
        -------
        numpy.ndarray
            The translation vector between the parent frame and this frame in the convention 0 with shape (3, 1).
        """
        T_dev = self.__T_dev.copy()
        T_dev.flags.writeable = True
        return T_dev
    
    @_T_dev.setter
    def _T_dev(self, T: numpy.ndarray) -> None:
        T = numpy.asarray(T).reshape((3,1)).astype(numpy.float64)
        if numpy.any(~numpy.isfinite(T)):
            raise ValueError("The translation must be finite.")
        self.__T_dev = T
        self.__T_dev.flags.writeable = False

    def _remove_parent_dev(self) -> None:
        r"""
        Remove the parent frame of the frame by setting the parent attribute to None.

        The R and T of the frame are changed to the global R and T of the frame (with the removed parent).

        Returns
        -------
        None
        """
        if self._parent is not None:
            R_global = self.get_global_rotation(convention=0)
            T_global = self.get_global_translation(convention=0)
            self._R_dev = R_global
            self._T_dev = T_global
            self._parent = None

    def _format_axes_dev(self, axes: Any) -> numpy.ndarray:
        r"""
        Format the axes to be a 3x3 matrix.
        
        Parameters
        ----------
        axes : Any
            The axes to format.

        Returns
        -------
        numpy.ndarray
            The formatted axes.
        """
        axes = numpy.asarray(axes).astype(numpy.float64)
        if not axes.shape == (3, 3):
            raise ValueError("The axes must be a 3x3 matrix.")
        norm = numpy.linalg.norm(axes, axis=0)
        if numpy.any(~numpy.isfinite(axes)):
            raise ValueError("The axes must be finite.")
        if numpy.any(numpy.isclose(norm, 0)):
            raise ValueError("The basis vectors must not be 0.")
        axes = axes / norm
        return axes

    def _format_rotation_dev(self, rotation: Any) -> Rotation:
        r"""
        Format the rotation to be a scipy.spatial.transform.Rotation object.

        Parameters
        ----------
        rotation : Any
            The rotation to format.

        Returns
        -------
        scipy.spatial.transform.Rotation
            The formatted rotation.
        """
        if not isinstance(rotation, scipy.spatial.transform.Rotation):
            raise TypeError("The rotation must be a scipy.spatial.transform.Rotation object.")
        return rotation
    
    def _format_translation_dev(self, translation: Any) -> numpy.ndarray:
        r"""
        Format the translation to be a 3x1 vector.

        Parameters
        ----------
        translation : Any
            The translation to format.

        Returns
        -------
        numpy.ndarray
            The formatted translation with shape (3, 1).
        """
        translation = numpy.asarray(translation).reshape((3,1)).astype(numpy.float64)
        if numpy.any(~numpy.isfinite(translation)):
            raise ValueError("The translation must be finite.")
        return translation
    
    def _format_rotation_matrix_dev(self, rotation_matrix: Any) -> numpy.ndarray:
        r"""
        Format the rotation matrix to be a 3x3 matrix.

        Parameters
        ----------
        rotation_matrix : Any
            The rotation matrix to format.

        Returns
        -------
        numpy.ndarray
            The formatted rotation matrix with shape (3, 3).
        """
        rotation_matrix = numpy.asarray(rotation_matrix).astype(numpy.float64)
        if numpy.any(~numpy.isfinite(rotation_matrix)):
            raise ValueError("The rotation matrix must be finite.")
        if not rotation_matrix.shape == (3, 3):
            raise ValueError("The rotation matrix must be a 3x3 matrix.")
        if not is_SO3(rotation_matrix):
            raise ValueError("The rotation matrix must be a special orthogonal matrix.")
        return rotation_matrix
    
    def _format_quaternion_dev(self, quaternion: Any) -> numpy.ndarray:
        r"""
        Format the quaternion to be a 4-element vector.

        Parameters
        ----------
        quaternion : Any
            The quaternion to format.

        Returns
        -------
        numpy.ndarray
            The formatted quaternion with shape (4,).
        """
        quaternion = numpy.asarray(quaternion).reshape((4,)).astype(numpy.float64)
        if numpy.any(~numpy.isfinite(quaternion)):
            raise ValueError("The quaternion must be finite.")
        norm = numpy.linalg.norm(quaternion)
        if numpy.isclose(norm, 0):
            raise ValueError("The quaternion must not be 0.")
        quaternion = quaternion / norm
        return quaternion
    
    def _format_euler_angles_dev(self, euler_angles: Any) -> numpy.ndarray:
        r"""
        Format the euler angles to be a 3-element vector.

        Parameters
        ----------
        euler_angles : Any
            The euler angles to format.

        Returns
        -------
        numpy.ndarray
            The formatted euler angles with shape (3,).
        """
        euler_angles = numpy.asarray(euler_angles).reshape((3,)).astype(numpy.float64)
        if numpy.any(~numpy.isfinite(euler_angles)):
            raise ValueError("The euler angles must be finite.")
        return euler_angles
    
    def _format_rotation_vector_dev(self, rotation_vector: Any) -> numpy.ndarray:
        r"""
        Format the rotation vector to be a 3-element vector.

        Parameters
        ----------
        rotation_vector : Any
            The rotation vector to format.

        Returns
        -------
        numpy.ndarray
            The formatted rotation vector with shape (3,).
        """
        rotation_vector = numpy.asarray(rotation_vector).reshape((3,)).astype(numpy.float64)
        if numpy.any(~numpy.isfinite(rotation_vector)):
            raise ValueError("The rotation vector must be finite.")
        return rotation_vector

    def _format_convention_dev(self, convention: Any, allow_None: bool = False) -> int:
        r"""
        Format the convention to be an integer between 0 and 7.

        Parameters
        ----------
        convention : Any
            The convention to format.

        allow_None : bool, optional
            If True, None is allowed and will be converted to the current convention of the frame.

        Returns
        -------
        int
            The formatted convention.
        """
        if convention is None and allow_None:
            convention = self._convention
        if not isinstance(convention, int):
            raise TypeError("The convention must be an integer.")
        if not convention in range(8):
            raise ValueError("The convention must be an integer between 0 and 7.")
        return convention


    # ====================================================================================================================================
    # Class methods to create a Frame object
    # ====================================================================================================================================
    @classmethod
    def canonical(
        cls,
        *,
        parent: Optional[Frame] = None,
        convention: Optional[int] = 0
        ) -> Frame:
        r"""
        Create the canonical frame of reference of :math:`\mathbb{R}^3`.

        The canonical frame is defined by its origin at the zero vector and its basis vectors as the standard basis vectors.
        
        Parameters
        ----------
        parent : Optional[Frame], optional
            The parent frame of the frame. Default is None - the global frame.

        convention : int, optional
            Integer in ``[0, 7]`` selecting the convention to express the transformation. Default is 0.

        Returns
        -------
        Frame
            The canonical frame of reference of :math:`\mathbb{R}^3`.
        
        Examples
        --------
        Lets create the canonical frame.

        .. code-block:: python

            from py3dframe import Frame

            canonical_frame = Frame.canonical()

        The canonical frame is defined by its origin at the zero vector and its basis vectors as the standard basis vectors.

        .. code-block:: python

            print("Origin of the canonical frame:", canonical_frame.origin)
            print("X-axis of the canonical frame:", canonical_frame.x_axis)
            print("Y-axis of the canonical frame:", canonical_frame.y_axis)
            print("Z-axis of the canonical frame:", canonical_frame.z_axis)
            # Output:
            # Origin of the canonical frame: [[0.] [0.] [0.]]
            # X-axis of the canonical frame: [[1.] [0.] [0.]]
            # Y-axis of the canonical frame: [[0.] [1.] [0.]]
            # Z-axis of the canonical frame: [[0.] [0.] [1.]]

        """
        return cls(parent=parent, convention=convention)
    
    @classmethod
    def from_rotation(
        cls,
        translation: Optional[numpy.ndarray] = None,
        rotation: Optional[scipy.spatial.transform.Rotation] = None,
        *,
        parent: Optional[Frame] = None,
        setup_only: bool = False,
        convention: Optional[int] = 0
        ) -> Frame:
        r"""
        Create a Frame object from a rotation (:class:`Rotation`) and a translation.

        .. seealso::

            - :class:`Rotation` class to create and manipulate rotations.
            - :meth:`from_rotation_matrix` method to create a Frame from a rotation matrix instead of a scipy Rotation object.
            - :meth:`from_quaternion` method to create a Frame from a quaternion instead of a scipy Rotation object.
            - :meth:`from_euler_angles` method to create a Frame from euler angles instead of a scipy Rotation object.
            - :meth:`from_rotation_vector` method to create a Frame from a rotation vector instead of a scipy Rotation object.
            - :meth:`from_axes` method to create a Frame from its axes (origin and basis vectors) instead of a rotation and a translation.

        Parameters
        ----------
        translation : numpy.ndarray, optional
            The translation vector of the transformation. It must be a 3 elements vector with shape (3, 1). Default is None - the zero vector.

        rotation : Rotation, optional
            The rotation of the transformation. It must be a Rotation. Default is None - the identity rotation.

        parent : Optional[Frame], optional
            The parent frame of the frame. Default is None - the global frame.

        setup_only : bool, optional
            If True, the parent frame will be used only to define the frame and not to link the frames. Default is False.

        convention : int, optional
            Integer in ``[0, 7]`` selecting the convention to express the transformation. Default is 0.

        Returns
        -------
        Frame
            The Frame object created from the given rotation and translation.

        Examples
        --------
        Lets create a frame from a rotation and a translation.

        We want to create the frame defined by the following origin and basis vectors in the parent frame coordinates:

        - Origin: :math:`O_F = [-1, -2, -3]`
        - X-axis: :math:`\mathbf{e}_1 = [1, 1, 0] / \sqrt{2}`
        - Y-axis: :math:`\mathbf{e}_2 = [-1, 1, 0] / \sqrt{2}`
        - Z-axis: :math:`\mathbf{e}_3 = [0, 0, 1]`

        With convention 0, the frame is defined by the following formula:

        .. math::

            \mathbf{X}_E = \mathbf{R} \mathbf{X}_F + \mathbf{T}

        For :math:`\mathbf{X}_F = 0` we have :math:`\mathbf{X}_E = O_F = \mathbf{T}` so the translation vector is directly the origin of the frame in the parent frame coordinates.
        For :math:`\mathbf{X}_F = [1, 0, 0]` we have :math:`\mathbf{X}_E = \mathbf{e}_1` so the first column of the rotation matrix is the x-axis of the frame in the parent frame coordinates.
        For :math:`\mathbf{X}_F = [0, 1, 0]` we have :math:`\mathbf{X}_E = \mathbf{e}_2` so the second column of the rotation matrix is the y-axis of the frame in the parent frame coordinates.
        For :math:`\mathbf{X}_F = [0, 0, 1]` we have :math:`\mathbf{X}_E = \mathbf{e}_3` so the third column of the rotation matrix is the z-axis of the frame in the parent frame coordinates.

        The frame can be created as follows:

        .. code-block:: python

            import numpy
            from py3dframe import Frame, Rotation

            origin = numpy.array([-1, -2, -3])
            x_axis = numpy.array([1, 1, 0]) / numpy.sqrt(2)
            y_axis = numpy.array([-1, 1, 0]) / numpy.sqrt(2)
            z_axis = numpy.array([0, 0, 1])

            axes = numpy.column_stack((x_axis, y_axis, z_axis))
            R = Rotation.from_matrix(axes)
            t = origin

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.from_rotation(translation=t, rotation=R, convention=0, parent=parent)

            print("Origin of the frame:", frame.origin)
            print("X-axis of the frame:", frame.x_axis)
            print("Y-axis of the frame:", frame.y_axis)
            print("Z-axis of the frame:", frame.z_axis)
            # Output:
            # Origin of the frame: [[-1.] [-2.] [-3.]]
            # X-axis of the frame: [[ 0.70710678] [ 0.70710678] [ 0.        ]]
            # Y-axis of the frame: [[-0.70710678] [ 0.70710678] [ 0.        ]]
            # Z-axis of the frame: [[0.] [0.] [1.]]

        If you like to work with a given convention you can specify it with the ``convention`` parameter.
        The translation and the rotation must be given in this convention to create the frame.

        You can also create the frame with convention 0 and then change the convention of the frame with the :attr`convention` attribute according to your preferences.

        """
        return cls(translation, rotation, parent=parent, setup_only=setup_only, convention=convention)


    @classmethod
    def from_axes(
        cls,
        origin: Optional[numpy.ndarray] = None,
        x_axis: Optional[numpy.ndarray] = None,
        y_axis: Optional[numpy.ndarray] = None,
        z_axis: Optional[numpy.ndarray] = None,
        *,
        parent: Optional[Frame] = None,
        setup_only: bool = False,
        convention: Optional[int] = 0
        ) -> Frame:
        r"""
        Create a Frame object from the given axes (origin and basis vectors).

        .. seealso::

            - :meth:`from_rotation` method to create a Frame from a rotation and a translation instead of its axes.
            - :meth:`from_rotation_matrix` method to create a Frame from a rotation matrix and a translation instead of its axes.
            - :meth:`from_quaternion` method to create a Frame from a quaternion and a translation instead of its axes.
            - :meth:`from_euler_angles` method to create a Frame from euler angles and a translation instead of its axes.
            - :meth:`from_rotation_vector` method to create a Frame from a rotation vector and a translation instead of its axes.

        Parameters
        ----------
        origin : array_like, optional
            The coordinates of the origin of the frame in the parent frame coordinates. Default is None - the zero vector.

        x_axis : array_like, optional
            The x-axis of the frame in the parent frame coordinates. Default is None - the [1, 0, 0] vector.

        y_axis : array_like, optional
            The y-axis of the frame in the parent frame coordinates. Default is None - the [0, 1, 0] vector.

        z_axis : array_like, optional
            The z-axis of the frame in the parent frame coordinates. Default is None - the [0, 0, 1] vector.

        parent : Optional[Frame], optional
            The parent frame of the frame. Default is None - the global frame.

        setup_only : bool, optional
            If True, the parent frame will be used only to define the frame and not to link the frames. Default is False.

        convention : int, optional
            Integer in ``[0, 7]`` selecting the convention to express the transformation. Default is 0.
        
        Returns
        -------
        Frame
            The Frame object created from the given axes.

        Examples
        --------
        Lets create a frame from a given origin and basis vectors.

        We want to create the frame defined by the following origin and basis vectors in the parent frame coordinates:

        - Origin: :math:`O_F = [-1, -2, -3]`
        - X-axis: :math:`\mathbf{e}_1 = [1, 1, 0] / \sqrt{2}`
        - Y-axis: :math:`\mathbf{e}_2 = [-1, 1, 0] / \sqrt{2}`
        - Z-axis: :math:`\mathbf{e}_3 = [0, 0, 1]`

        The frame can be created as follows:

        .. code-block:: python

            import numpy
            from py3dframe import Frame

            origin = numpy.array([-1, -2, -3])
            x_axis = numpy.array([1, 1, 0]) / numpy.sqrt(2)
            y_axis = numpy.array([-1, 1, 0]) / numpy.sqrt(2)
            z_axis = numpy.array([0, 0, 1])

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.from_axes(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, parent=parent)

            print("Origin of the frame:", frame.origin)
            print("X-axis of the frame:", frame.x_axis)
            print("Y-axis of the frame:", frame.y_axis)
            print("Z-axis of the frame:", frame.z_axis)
            # Output:
            # Origin of the frame: [[-1.] [-2.] [-3.]]
            # X-axis of the frame: [[ 0.70710678] [ 0.70710678] [ 0.        ]]
            # Y-axis of the frame: [[-0.70710678] [ 0.70710678] [ 0.        ]]
            # Z-axis of the frame: [[0.] [0.] [1.]]

        If you like to work with a given convention, you can specify it with the ``convention`` parameter in order to access the translation and rotation in this convention.

        """
        # Set default values
        origin = origin if origin is not None else [0, 0, 0]
        x_axis = x_axis if x_axis is not None else [1, 0, 0]
        y_axis = y_axis if y_axis is not None else [0, 1, 0]
        z_axis = z_axis if z_axis is not None else [0, 0, 1]
        # Construct the frame
        axes = numpy.column_stack((x_axis, y_axis, z_axis)).astype(numpy.float64)
        instance = cls.canonical(parent=parent, convention=convention)
        instance.origin = origin
        instance.axes = axes
        if setup_only:
            instance._remove_parent_dev()
        return instance


    @classmethod
    def from_rotation_matrix(
        cls,
        translation: Optional[numpy.ndarray] = None,
        rotation_matrix: Optional[numpy.ndarray] = None,
        *,
        parent: Optional[Frame] = None,
        setup_only: bool = False,
        convention: Optional[int] = 0
        ) -> Frame:
        r"""
        Create a Frame object from a rotation matrix and translation.

        .. seealso::

            - :meth:`from_rotation` method to create a Frame from a Rotation object instead of a rotation matrix.
            - :meth:`from_quaternion` method to create a Frame from a quaternion instead of a rotation matrix.
            - :meth:`from_euler_angles` method to create a Frame from euler angles instead of a rotation matrix.
            - :meth:`from_rotation_vector` method to create a Frame from a rotation vector instead of a rotation matrix.
            - :meth:`from_axes` method to create a Frame from its axes (origin and basis vectors) instead of a rotation matrix and a translation.

        Parameters
        ----------
        translation : numpy.ndarray, optional
            The translation vector of the transformation. It must be a 3 elements vector with shape (3, 1). Default is None - the zero vector.

        rotation_matrix : numpy.ndarray, optional
            The rotation matrix of the transformation. It must be a 3x3 matrix. Default is None - the identity matrix.

        parent : Optional[Frame], optional
            The parent frame of the frame. Default is None - the global frame.

        setup_only : bool, optional
            If True, the parent frame will be used only to define the frame and not to link the frames. Default is False.

        convention : int, optional
            Integer in ``[0, 7]`` selecting the convention to express the transformation. Default is 0.

        Returns
        -------
        Frame
            The Frame object created from the given rotation matrix and translation.

        Examples
        --------
        Lets create a frame from a rotation matrix and a translation.

        We want to create the frame defined by the following origin and basis vectors in the parent frame coordinates:

        - Origin: :math:`O_F = [-1, -2, -3]`
        - X-axis: :math:`\mathbf{e}_1 = [1, 1, 0] / \sqrt{2}`
        - Y-axis: :math:`\mathbf{e}_2 = [-1, 1, 0] / \sqrt{2}`
        - Z-axis: :math:`\mathbf{e}_3 = [0, 0, 1]`

        With convention 0, the frame is defined by the following formula:

        .. math::

            \mathbf{X}_E = \mathbf{R} \mathbf{X}_F + \mathbf{T}

        For :math:`\mathbf{X}_F = 0` we have :math:`\mathbf{X}_E = O_F = \mathbf{T}` so the translation vector is directly the origin of the frame in the parent frame coordinates.
        For :math:`\mathbf{X}_F = [1, 0, 0]` we have :math:`\mathbf{X}_E = \mathbf{e}_1` so the first column of the rotation matrix is the x-axis of the frame in the parent frame coordinates.
        For :math:`\mathbf{X}_F = [0, 1, 0]` we have :math:`\mathbf{X}_E = \mathbf{e}_2` so the second column of the rotation matrix is the y-axis of the frame in the parent frame coordinates.
        For :math:`\mathbf{X}_F = [0, 0, 1]` we have :math:`\mathbf{X}_E = \mathbf{e}_3` so the third column of the rotation matrix is the z-axis of the frame in the parent frame coordinates.

        The frame can be created as follows:

        .. code-block:: python

            import numpy
            from py3dframe import Frame

            origin = numpy.array([-1, -2, -3])
            x_axis = numpy.array([1, 1, 0]) / numpy.sqrt(2)
            y_axis = numpy.array([-1, 1, 0]) / numpy.sqrt(2)
            z_axis = numpy.array([0, 0, 1])

            axes = numpy.column_stack((x_axis, y_axis, z_axis))
            R = axes
            t = origin

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.from_rotation_matrix(translation=t, rotation_matrix=R, convention=0, parent=parent)

            print("Origin of the frame:", frame.origin)
            print("X-axis of the frame:", frame.x_axis)
            print("Y-axis of the frame:", frame.y_axis)
            print("Z-axis of the frame:", frame.z_axis)
            # Output:
            # Origin of the frame: [[-1.] [-2.] [-3.]]
            # X-axis of the frame: [[ 0.70710678] [ 0.70710678] [ 0.        ]]
            # Y-axis of the frame: [[-0.70710678] [ 0.70710678] [ 0.        ]]
            # Z-axis of the frame: [[0.] [0.] [1.]]

        If you like to work with a given convention you can specify it with the ``convention`` parameter.
        The translation and the rotation matrix must be given in this convention to create the frame.

        You can also create the frame with convention 0 and then change the convention of the frame with the :attr`convention` attribute according to your preferences.

        """
        # Set default values
        rotation_matrix = rotation_matrix if rotation_matrix is not None else numpy.eye(3)
        translation = translation if translation is not None else numpy.zeros((3,1), dtype=numpy.float64)
        # Construct the frame
        instance = cls.canonical(parent=parent, convention=convention)
        instance.set_translation(translation, convention=convention)
        instance.set_rotation_matrix(rotation_matrix, convention=convention)
        if setup_only:
            instance._remove_parent_dev()
        return instance
    

    @classmethod
    def from_quaternion(
        cls,
        translation: Optional[numpy.ndarray] = None,
        quaternion: Optional[numpy.ndarray] = None,
        *,
        parent: Optional[Frame] = None,
        setup_only: bool = False,
        convention: Optional[int] = 0,
        scalar_first: bool = True
        ) -> Frame:
        r"""
        Create a Frame object from a quaternion and translation.

        The quaternion must be in the [w, x, y, z] format if ``scalar_first`` is True and in the [x, y, z, w] format if ``scalar_first`` is False.

        .. seealso::

            - :meth:`from_rotation` method to create a Frame from a Rotation object instead of a quaternion.
            - :meth:`from_rotation_matrix` method to create a Frame from a rotation matrix instead of a quaternion.
            - :meth:`from_euler_angles` method to create a Frame from euler angles instead of a quaternion.
            - :meth:`from_rotation_vector` method to create a Frame from a rotation vector instead of a quaternion.
            - :meth:`from_axes` method to create a Frame from its axes (origin and basis vectors) instead of a quaternion and a translation

        Parameters
        ----------
        translation : numpy.ndarray, optional
            The translation vector of the transformation. It must be a 3 elements vector with shape (3, 1). Default is None - the zero vector.

        quaternion : numpy.ndarray, optional
            The quaternion of the transformation. It must be a 4 elements vector [w, x, y, z] (scalar first convention). Default is None - the identity quaternion.

        parent : Optional[Frame], optional
            The parent frame of the frame. Default is None - the global frame.

        setup_only : bool, optional
            If True, the parent frame will be used only to define the frame and not to link the frames. Default is False.

        convention : int, optional
            Integer in ``[0, 7]`` selecting the convention to express the transformation. Default is 0.

        scalar_first : bool, optional
            If True, the quaternion is in the [w, x, y, z] format. If False, the quaternion is in the [x, y, z, w] format. Default is True.

        Returns
        -------
        Frame
            The Frame object created from the given quaternion and translation.

        Examples
        --------
        Lets create a frame from a quaternion and a translation.

        We want to create the frame defined by the following origin and basis vectors in the parent frame coordinates:

        - Origin: :math:`O_F = [-1, -2, -3]`
        - X-axis: :math:`\mathbf{e}_1 = [1, 1, 0] / \sqrt{2}`
        - Y-axis: :math:`\mathbf{e}_2 = [-1, 1, 0] / \sqrt{2}`
        - Z-axis: :math:`\mathbf{e}_3 = [0, 0, 1]`

        With convention 0, the frame is defined by the following formula:

        .. math::

            \mathbf{X}_E = \mathbf{R} \mathbf{X}_F + \mathbf{T}

        For :math:`\mathbf{X}_F = 0` we have :math:`\mathbf{X}_E = O_F = \mathbf{T}` so the translation vector is directly the origin of the frame in the parent frame coordinates.
        The rotation matrix corresponding to the basis vectors in the convention 0 can be described with quaternion :math:`q = [w, x, y, z] = [0.5 \sqrt{2 + \sqrt{2}}, 0, 0, \sqrt{2}/(4 w)]`.

        .. code-block:: python

            import numpy
            from py3dframe import Frame

            w = 0.5 * numpy.sqrt(2 + numpy.sqrt(2))
            z = numpy.sqrt(2) / (4 * w)
            quaternion = numpy.array([w, 0, 0, z]) # [w, x, y, z]
            origin = numpy.array([-1, -2, -3])
            t = origin

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.from_quaternion(translation=t, quaternion=quaternion, convention=0, parent=parent)

            print("Origin of the frame:", frame.origin)
            print("X-axis of the frame:", frame.x_axis)
            print("Y-axis of the frame:", frame.y_axis)
            print("Z-axis of the frame:", frame.z_axis)
            # Output:
            # Origin of the frame: [[-1.] [-2.] [-3.]]
            # X-axis of the frame: [[ 0.70710678] [ 0.70710678] [ 0.        ]]
            # Y-axis of the frame: [[-0.70710678] [ 0.70710678] [ 0.        ]]
            # Z-axis of the frame: [[0.] [0.] [1.]]

        If you like to work with a given convention you can specify it with the ``convention`` parameter.
        The translation and the quaternion must be given in this convention to create the frame.

        You can also create the frame with convention 0 and then change the convention of the frame with the :attr`convention` attribute according to your preferences.        

        """
        # Set default values
        quaternion = quaternion if quaternion is not None else numpy.array([1, 0, 0, 0]) # Identity quaternion [w, x, y, z]
        translation = translation if translation is not None else numpy.zeros((3,1), dtype=numpy.float64)
        # Construct the frame
        instance = cls.canonical(parent=parent, convention=convention)
        instance.set_translation(translation, convention=convention)
        instance.set_quaternion(quaternion, convention=convention, scalar_first=scalar_first)
        if setup_only:
            instance._remove_parent_dev()
        return instance
    

    @classmethod
    def from_euler_angles(
        cls,
        translation: Optional[numpy.ndarray] = None,
        euler_angles: Optional[numpy.ndarray] = None,
        *,
        parent: Optional[Frame] = None,
        setup_only: bool = False,
        convention: Optional[int] = 0,
        degrees: bool = False,
        seq: str = "xyz"
        ) -> Frame:
        r"""
        Create a Frame object from euler angles and translation.

        The euler angles allows to represent any rotation by a sequence of three elemental rotations around the axes of a coordinate system.
        The sequence of the rotations must be specified with the ``seq`` parameter.

        .. seealso::

            - :meth:`from_rotation` method to create a Frame from a Rotation object instead of euler angles.
            - :meth:`from_rotation_matrix` method to create a Frame from a rotation matrix instead of euler angles.
            - :meth:`from_quaternion` method to create a Frame from a quaternion instead of euler angles.
            - :meth:`from_rotation_vector` method to create a Frame from a rotation vector instead of euler angles.
            - :meth:`from_axes` method to create a Frame from its axes (origin and basis vectors) instead of euler angles and a translation.

        Parameters
        ----------
        translation : numpy.ndarray, optional
            The translation vector of the transformation. It must be a 3 elements vector with shape (3, 1). Default is None - the zero vector.

        euler_angles : numpy.ndarray, optional
            The euler angles of the transformation. It must be a 3 elements vector [alpha, beta, gamma] in radians with xyz convention. Default is None - the zero vector.

        parent : Optional[Frame], optional
            The parent frame of the frame. Default is None - the global frame.

        setup_only : bool, optional
            If True, the parent frame will be used only to define the frame and not to link the frames. Default is False.

        convention : int, optional
            Integer in ``[0, 7]`` selecting the convention to express the transformation. Default is 0.

        degrees : bool, optional
            If True, the euler angles are in degrees. If False, the euler angles are in radians. Default is False.

        seq : str, optional
            The sequence of the euler angles. It must be a string of 3 characters chosen between 'x', 'y', and 'z'. Default is "xyz".

        Returns
        -------
        Frame
            The Frame object created from the given euler angles and translation.

        Examples
        --------
        Lets create a frame from euler angles and a translation.

        We want to create the frame defined by the following origin and basis vectors in the parent frame coordinates:

        - Origin: :math:`O_F = [-1, -2, -3]`
        - X-axis: :math:`\mathbf{e}_1 = [1, 1, 0] / \sqrt{2}`
        - Y-axis: :math:`\mathbf{e}_2 = [-1, 1, 0] / \sqrt{2}`
        - Z-axis: :math:`\mathbf{e}_3 = [0, 0, 1]`

        With convention 0, the frame is defined by the following formula:

        .. math::

            \mathbf{X}_E = \mathbf{R} \mathbf{X}_F + \mathbf{T}

        For :math:`\mathbf{X}_F = 0` we have :math:`\mathbf{X}_E = O_F = \mathbf{T}` so the translation vector is directly the origin of the frame in the parent frame coordinates.
        The rotation matrix corresponding to the basis vectors in the convention 0 can be described with euler angles :math:`[\alpha, \beta, \gamma] = [0, 0, \pi/4]` in the 'xyz' sequence.

        .. code-block:: python

            import numpy
            from py3dframe import Frame

            euler_angles = numpy.array([0, 0, numpy.pi/4]) # in radians
            origin = numpy.array([-1, -2, -3])
            t = origin

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.from_euler_angles(translation=t, euler_angles=euler_angles, convention=0, parent=parent)

            print("Origin of the frame:", frame.origin)
            print("X-axis of the frame:", frame.x_axis)
            print("Y-axis of the frame:", frame.y_axis)
            print("Z-axis of the frame:", frame.z_axis)
            # Output:
            # Origin of the frame: [[-1.] [-2.] [-3.]]
            # X-axis of the frame: [[ 0.70710678] [ 0.70710678] [ 0.        ]]
            # Y-axis of the frame: [[-0.70710678] [ 0.70710678] [ 0.        ]]
            # Z-axis of the frame: [[0.] [0.] [1.]]

        If you like to work with a given convention you can specify it with the ``convention`` parameter.
        The translation and the euler angles must be given in this convention to create the frame.

        You can also create the frame with convention 0 and then change the convention of the frame with the :attr`convention` attribute according to your preferences.

        """
        # Set default values
        euler_angles = euler_angles if euler_angles is not None else numpy.array([0, 0, 0]) # Zero euler angles
        translation = translation if translation is not None else numpy.zeros((3,1), dtype=numpy.float64)
        # Construct the frame
        instance = cls.canonical(parent=parent, convention=convention)
        instance.set_translation(translation, convention=convention)
        instance.set_euler_angles(euler_angles, convention=convention, degrees=degrees, seq=seq)
        if setup_only:
            instance._remove_parent_dev()
        return instance
    
    

    @classmethod
    def from_rotation_vector(
        cls,
        translation: Optional[numpy.ndarray] = None,
        rotation_vector: Optional[numpy.ndarray] = None,
        *,
        parent: Optional[Frame] = None,
        setup_only: bool = False,
        convention: Optional[int] = 0,
        degrees: bool = False
        ) -> Frame:
        r"""
        Create a Frame object from a rotation vector and translation.

        A rotation vector is a 3D vector that represents a rotation in 3D space. The direction of the vector indicates the axis of rotation, and the magnitude (length) of the vector indicates the angle of rotation in radians.

        .. seealso::

            - :meth:`from_rotation` method to create a Frame from a Rotation object instead of a rotation vector.
            - :meth:`from_rotation_matrix` method to create a Frame from a rotation matrix instead of a rotation vector.
            - :meth:`from_quaternion` method to create a Frame from a quaternion instead of a rotation vector.
            - :meth:`from_euler_angles` method to create a Frame from euler angles instead of a rotation vector.
            - :meth:`from_axes` method to create a Frame from its axes (origin and basis vectors) instead of a rotation vector and a translation.

        Parameters
        ----------
        translation : numpy.ndarray, optional
            The translation vector of the transformation. It must be a 3 elements vector with shape (3, 1). Default is None - the zero vector.

        rotation_vector : numpy.ndarray, optional
            The rotation vector of the transformation. It must be a 3 elements vector in radians. Default is None - the zero vector.

        parent : Optional[Frame], optional
            The parent frame of the frame. Default is None - the global frame.

        setup_only : bool, optional
            If True, the parent frame will be used only to define the frame and not to link the frames. Default is False.

        convention : int, optional
            Integer in ``[0, 7]`` selecting the convention to express the transformation. Default is 0.

        degrees : bool, optional
            If True, the rotation vector is in degrees. If False, the rotation vector is in radians. Default is False.

        Returns
        -------
        Frame
            The Frame object created from the given rotation vector and translation.
        
        Examples
        --------
        Lets create a frame from a rotation vector and a translation.

        We want to create the frame defined by the following origin and basis vectors in the parent frame coordinates:

        - Origin: :math:`O_F = [-1, -2, -3]`
        - X-axis: :math:`\mathbf{e}_1 = [1, 1, 0] / \sqrt{2}`
        - Y-axis: :math:`\mathbf{e}_2 = [-1, 1, 0] / \sqrt{2}`
        - Z-axis: :math:`\mathbf{e}_3 = [0, 0, 1]`

        With convention 0, the frame is defined by the following formula:

        .. math::

            \mathbf{X}_E = \mathbf{R} \mathbf{X}_F + \mathbf{T}

        For :math:`\mathbf{X}_F = 0` we have :math:`\mathbf{X}_E = O_F = \mathbf{T}` so the translation vector is directly the origin of the frame in the parent frame coordinates.
        The rotation matrix corresponding to the basis vectors in the convention 0 can be described with rotation vector :math:`\mathbf{r} = [0, 0, \pi/4]`.

        .. code-block:: python

            import numpy
            from py3dframe import Frame

            rotation_vector = numpy.array([0, 0, numpy.pi/4]) # in radians
            origin = numpy.array([-1, -2, -3])
            t = origin

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.from_rotation_vector(translation=t, rotation_vector=rotation_vector, convention=0, parent=parent)

            print("Origin of the frame:", frame.origin)
            print("X-axis of the frame:", frame.x_axis)
            print("Y-axis of the frame:", frame.y_axis)
            print("Z-axis of the frame:", frame.z_axis)
            # Output:
            # Origin of the frame: [[-1.] [-2.] [-3.]]
            # X-axis of the frame: [[ 0.70710678] [ 0.70710678] [ 0.        ]]
            # Y-axis of the frame: [[-0.70710678] [ 0.70710678] [ 0.        ]]
            # Z-axis of the frame: [[0.] [0.] [1.]]

        If you like to work with a given convention you can specify it with the ``convention`` parameter.
        The translation and the rotation vector must be given in this convention to create the frame.

        You can also create the frame with convention 0 and then change the convention of the frame with the :attr`convention` attribute according to your preferences.

        """
        # Set default values
        rotation_vector = rotation_vector if rotation_vector is not None else numpy.array([0, 0, 0]) # Zero rotation vector
        translation = translation if translation is not None else numpy.zeros((3,1), dtype=numpy.float64)
        # Construct the frame
        instance = cls.canonical(parent=parent, convention=convention)
        instance.set_translation(translation, convention=convention)
        instance.set_rotation_vector(rotation_vector, convention=convention, degrees=degrees)
        if setup_only:
            instance._remove_parent_dev()
        return instance

    
    # ====================================================================================================================================
    # User methods
    # ====================================================================================================================================
    @property
    def parent(self) -> Optional[Frame]:
        r"""
        The parent frame of this frame (by default the canonical frame if None).

        .. note::

            This property is settable.

        The parent frame is used to define this frame relatively this parent frame.

        .. warning::

            If you change the parent frame of a frame, the current transformation (translation and rotation) of the frame will be kept unchanged but the global transformation (global translation and global rotation) of the frame will change according to the new parent frame.

        Parameters
        ----------
        parent : Optional[Frame]
            The parent frame in which the frame will be defined. If None, the frame will be defined in the global frame (the canonical frame).

        Returns
        -------
        Optional[Frame]
            The parent frame of the frame. If None, the frame is defined in the global frame (the canonical frame).

        Examples
        --------
        Lets consider a train and a person standing into the train.

        First we can define the train frame relative to the global frame (the canonical frame) and then we can define the person frame relative to the train frame.

        .. code-block:: python

            from py3dframe import Frame

            train = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=None)
            person = Frame.from_axes(origin=[0, 1, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=train)

        If the train moves and the person stays immobile in the train, the person axis will be unchanged relative to the train frame but will move relative to the global frame.

        .. code-block:: python

            train.origin = [10, 0, 0] # The train moves 10 units along the x-axis of the global frame.

            print("Person origin in the parent frame:", person.origin)
            # Output: Person origin in the parent frame: [[ 0.] [ 1.] [ 0.]]

            print("Person origin in the global frame:", person.global_origin)
            # Output: Person origin in the global frame: [[10.] [ 1.] [ 0.]]

        The parent frame allows to define a hierarchy of frames.

        """
        return self._parent
    
    @parent.setter
    def parent(self, parent: Optional[Frame]) -> None:
        if parent is not None and not isinstance(parent, Frame):
            raise TypeError("The parent must be a Frame object.")
        self._parent = parent
    


    @property
    def convention(self) -> int:
        r"""
        The convention to express the transformation between the parent frame and this frame.

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

        - :math:`\mathbf{X}_E` is a point expressed in the parent frame coordinates.
        - :math:`\mathbf{X}_F` is the same point expressed in the frame coordinates.
        - :math:`\mathbf{R}` is the rotation matrix between the parent frame and this frame.
        - :math:`\mathbf{T}` is the translation vector between the parent frame and this frame.

        .. note::

            The default convention is 0.

        This can be useful when working with different libraries or applications that use different conventions for representing 3D transformations (as OPENCV, OPENGL, ROS, ...).

        Parameters
        ----------
        convention : int
            Integer in ``[0, 7]`` selecting the convention to express the transformation between the parent frame and this frame.

        Returns
        -------
        int
            The convention parameter.

        Examples
        --------
        Lets create a frame with given axes and origin with the default convention 0.

        .. code-block:: python

            import numpy
            from py3dframe import Frame

            origin = numpy.array([-1, -2, -3])
            x_axis = numpy.array([1, 1, 0]) / numpy.sqrt(2)
            y_axis = numpy.array([-1, 1, 0]) / numpy.sqrt(2)
            z_axis = numpy.array([0, 0, 1])

            frame = Frame.from_axes(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, convention=0)

        If an application uses another convention, you can access the translation and rotation in this convention by changing the convention of the frame.

        .. code-block:: python

            frame.convention = 4 # Change the convention of the frame to convention 4.

            print("Rotation matrix in convention 4:", frame.rotation_matrix)

        The rotation returned is the rotation between the parent frame and this frame in convention 4.

        """
        return self._convention
    
    @convention.setter
    def convention(self, convention: int) -> None:
        convention = self._format_convention_dev(convention, allow_None=False)
        self._convention = convention
    
    @property
    def origin(self) -> numpy.ndarray:
        r"""
        The origin of the frame relative to the parent frame.

        The origin is a 3 elements vector with shape (3, 1) representing the coordinates of the origin of the frame in the parent frame coordinates.

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :meth:`global_origin` to access the origin of the frame in the global frame coordinates.

        Parameters
        ----------
        origin : numpy.ndarray
            The origin of the frame in the parent frame coordinates as an array-like with 3 elements.

        Returns
        -------
        numpy.ndarray
            The origin of the frame in the parent frame coordinates with shape (3, 1).

        """
        return self.get_translation(convention=0)
    
    @origin.setter
    def origin(self, origin: numpy.ndarray) -> None:
        self.set_translation(origin, convention=0)
    

    @property
    def axes(self) -> numpy.ndarray:
        r"""
        The basis vectors of the frame relative to the parent frame.

        The axes is a 3x3 matrix with shape (3, 3) representing the basis vectors of the frame in the parent frame coordinates.
        The first column is the x-axis, the second column is the y-axis and the third column is the z-axis.

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`x_axis` to access the x-axis of the frame.
            - attribute :attr:`y_axis` to access the y-axis of the frame.
            - attribute :attr:`z_axis` to access the z-axis of the frame.
            - attribute :attr:`global_axes` to access the basis vectors of the frame in the global frame coordinates.

        Parameters
        ----------
        axes : numpy.ndarray
            The basis vectors of the frame in the parent frame coordinates as an array-like with shape (3, 3).

        Returns
        -------
        numpy.ndarray
            The basis vectors of the frame in the parent frame coordinates with shape (3, 3).

        
        Examples
        --------

        Lets define the axes of a frame.

            import numpy
            from py3dframe import Frame

            frame = Frame.canonical()

            # Define the axes of the frame (Axes will be normalized)
            x_axis = numpy.array([1, 0, 0])
            y_axis = numpy.array([0, 2, 0])
            z_axis = numpy.array([0, 0, 3])

            axes = numpy.column_stack((x_axis, y_axis, z_axis))
            frame.axes = axes
            print("Axes of the frame:", frame.axes)
            # Output: Axes of the frame: [[1. 0. 0.]
            #                          [0. 1. 0.]
            #                          [0. 0. 1.]]    

        """
        return self.get_rotation_matrix(convention=0)
    
    @axes.setter
    def axes(self, axes: numpy.ndarray) -> None:
        axes = self._format_axes_dev(axes)
        self.set_rotation_matrix(axes, convention=0)
    

    @property
    def x_axis(self) -> numpy.ndarray:
        r"""
        The x-axis of the frame relative to the parent frame.

        The x-axis is a 3 elements vector with shape (3, 1) representing the coordinates of the x-axis of the frame in the parent frame coordinates.

        .. note::

            This property is not settable. To change the x-axis, use the :attr:`axes` attribute.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`y_axis` to access the y-axis of the frame.
            - attribute :attr:`z_axis` to access the z-axis of the frame.
            - attribute :attr:`global_x_axis` to access the x-axis of the frame in the global frame coordinates.

        Returns
        -------
        numpy.ndarray
            The x-axis of the frame in the parent frame coordinates with shape (3, 1).
        """
        x_axis = self.axes[:,0].reshape((3,1))
        return x_axis
    

    @property
    def y_axis(self) -> numpy.ndarray:
        r"""
        The y-axis of the frame relative to the parent frame.

        The y-axis is a 3 elements vector with shape (3, 1) representing the coordinates of the y-axis of the frame in the parent frame coordinates.

        .. note::

            This property is not settable. To change the y-axis, use the :attr:`axes` attribute.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`x_axis` to access the x-axis of the frame.
            - attribute :attr:`z_axis` to access the z-axis of the frame.
            - attribute :attr:`global_y_axis` to access the y-axis of the frame in the global frame coordinates.

        Returns
        -------
        numpy.ndarray
            The y-axis of the frame in the parent frame coordinates with shape (3, 1).
        """
        y_axis = self.axes[:,1].reshape((3,1))
        return y_axis
    

    @property
    def z_axis(self) -> numpy.ndarray:
        r"""
        The z-axis of the frame relative to the parent frame.

        The z-axis is a 3 elements vector with shape (3, 1) representing the coordinates of the z-axis of the frame in the parent frame coordinates.

        .. note::

            This property is not settable. To change the z-axis, use the :attr:`axes` attribute.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`x_axis` to access the x-axis of the frame.
            - attribute :attr:`y_axis` to access the y-axis of the frame.
            - attribute :attr:`global_z_axis` to access the z-axis of the frame in the global frame coordinates.

        Returns
        -------
        numpy.ndarray
            The z-axis of the frame in the parent frame coordinates with shape (3, 1).
        """
        z_axis = self.axes[:,2].reshape((3,1))
        return z_axis


    def get_rotation(self, *, convention: Optional[int] = None) -> scipy.spatial.transform.Rotation:
        r"""
        Access the rotation between the parent frame and this frame in a specific convention.

        The rotation is returned as a :class:`Rotation` object.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`rotation` to get or set the rotation in the convention of the frame.
            - method :meth:`set_rotation` to set the rotation in a specific convention.
            - method :meth:`get_global_rotation` to get the rotation between the global frame and this frame.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        Returns
        -------
        Rotation
            The rotation between the parent frame and this frame in the given convention.

        Examples
        --------
        Lets create a frame from its axes and origin with the default convention 0.

        .. code-block:: python

            import numpy
            from py3dframe import Frame

            origin = numpy.array([-1, -2, -3])
            x_axis = numpy.array([1, 1, 0]) / numpy.sqrt(2)
            y_axis = numpy.array([-1, 1, 0]) / numpy.sqrt(2)
            z_axis = numpy.array([0, 0, 1])

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.from_axes(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, convention=0, parent=parent)

        If an application using an other convention required the rotation in the convention 4, you can access the rotation in this convention with the :meth:`get_rotation` method.

        .. code-block:: python

            rotation_convention_4 = frame.get_rotation(convention=4)

        """
        convention = self._format_convention_dev(convention, allow_None=True)
        R, _ = switch_RT_convention(self._R_dev, self._T_dev, 0, convention)
        return R
    
    def set_rotation(self, rotation: Rotation, *, convention: Optional[int] = None) -> None:
        r"""
        Set the rotation between the parent frame and this frame in a specific convention.

        The rotation must be given as a :class:`Rotation` object.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`rotation` to get or set the rotation in the convention of the frame.
            - method :meth:`get_rotation` to get the rotation in a specific convention.
            - method :meth:`set_global_rotation` to set the rotation between the global frame and this frame.

        Parameters
        ----------
        rotation : Rotation
            The rotation between the parent frame and this frame in the given convention.
        
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        Examples
        --------
        Lets create a default frame with convention 0.

        .. code-block:: python  

            from py3dframe import Frame
        
            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.canonical(convention=0, parent=parent)

        Lets assume, an application uses convention 4 to represent the transformation between two frames.
        The frame of reference of the application rotates and the new rotation between the parent frame and this frame in convention 4 can be extracted from the application.

        .. code-block:: python

            import numpy
            from py3dframe import Rotation

            rotation = Rotation.from_matrix(...) # Extract the rotation matrix from the application.

            frame.set_rotation(rotation, convention=4) # Set the rotation in convention 4.

        """
        convention = self._format_convention_dev(convention, allow_None=True)
        rotation = self._format_rotation_dev(rotation)
        _, current_T = switch_RT_convention(self._R_dev, self._T_dev, 0, convention)
        self._R_dev, self._T_dev = switch_RT_convention(rotation, current_T, convention, 0)

    @property
    def rotation(self) -> Rotation:
        r"""
        The rotation between the parent frame and this frame in the convention of the frame.

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - method :attr:`get_rotation` to get the rotation in a specific convention.
            - method :meth:`set_rotation` to set the rotation in a specific convention.

        Parameters
        ----------
        rotation : Rotation
            The rotation between the parent frame and this frame in the convention of the frame.

        Returns
        -------
        Rotation
            The rotation between the parent frame and this frame in the convention of the frame.
        """
        return self.get_rotation(convention=self._convention)
    
    @rotation.setter
    def rotation(self, rotation: scipy.spatial.transform.Rotation) -> None:
        self.set_rotation(rotation, convention=self._convention)
    

    def get_translation(self, *, convention: Optional[int] = None) -> numpy.ndarray:
        r"""
        Access the translation vector between the parent frame and this frame in a specific convention.

        The translation vector is returned as a numpy array with shape (3, 1).

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`translation` to get or set the translation in the convention of the frame.
            - method :meth:`set_translation` to set the translation in a specific convention.
            - method :meth:`get_global_translation` to get the translation between the global frame and this frame.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        Returns
        -------
        numpy.ndarray
            The translation vector between the parent frame and this frame in the given convention with shape (3, 1).

        Examples
        --------
        Lets create a frame from its axes and origin with the default convention 0.

        .. code-block:: python

            import numpy
            from py3dframe import Frame

            origin = numpy.array([-1, -2, -3])
            x_axis = numpy.array([1, 1, 0]) / numpy.sqrt(2)
            y_axis = numpy.array([-1, 1, 0]) / numpy.sqrt(2)
            z_axis = numpy.array([0, 0, 1])

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.from_axes(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, convention=0, parent=parent)

        If an application using an other convention required the translation in the convention 4, you can access the translation in this convention with the :meth:`get_translation` method.

        .. code-block:: python

            translation_convention_4 = frame.get_translation(convention=4)

        """
        convention = self._format_convention_dev(convention, allow_None=True)
        _, T = switch_RT_convention(self._R_dev, self._T_dev, 0, convention)
        return T
    
    def set_translation(self, translation: numpy.ndarray, *, convention: Optional[int] = None) -> None:
        r"""
        Set the translation vector between the parent frame and this frame in a specific convention.

        The translation vector must be given as a array-like with 3 elements.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`translation` to get or set the translation in the convention of the frame.
            - method :meth:`get_translation` to get the translation in a specific convention.
            - method :meth:`set_global_translation` to set the translation between the global frame and this frame.

        Parameters
        ----------
        translation : numpy.ndarray
            The translation vector between the parent frame and this frame in the given convention as an array-like with 3 elements.

        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        Examples
        --------
        Lets create a default frame with convention 0.

        .. code-block:: python

            from py3dframe import Frame

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.canonical(convention=0, parent=parent)

        Lets assume, an application uses convention 4 to represent the transformation between two frames.
        The frame of reference of the application moves and the new translation vector between the parent frame and this frame in convention 4 can be extracted from the application.

        .. code-block:: python

            import numpy

            translation = numpy.array(...) # Extract the translation vector from the application.

            frame.set_translation(translation, convention=4) # Set the translation in convention 4.

        """
        convention = self._format_convention_dev(convention, allow_None=True)
        translation = self._format_translation_dev(translation)
        current_R, _ = switch_RT_convention(self._R_dev, self._T_dev, 0, convention)
        self._R_dev, self._T_dev = switch_RT_convention(current_R, translation, convention, 0)

    @property
    def translation(self) -> numpy.ndarray:
        r"""
        The translation vector between the parent frame and this frame in the convention of the frame.

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - method :meth:`get_translation` to get the translation in a specific convention.
            - method :meth:`set_translation` to set the translation in a specific convention.

        Parameters
        ----------
        translation : numpy.ndarray
            The translation vector between the parent frame and this frame in the convention of the frame as an array-like with 3 elements.

        Returns
        -------
        numpy.ndarray
            The translation vector between the parent frame and this frame in the convention of the frame with shape (3, 1).

        """
        return self.get_translation(convention=self._convention)
    
    @translation.setter
    def translation(self, translation: numpy.ndarray) -> None:
        self.set_translation(translation, convention=self._convention)
    

    def get_rotation_matrix(self, *, convention: Optional[int] = None) -> numpy.ndarray:
        r"""
        Access the rotation matrix between the parent frame and this frame in a specific convention.

        The rotation matrix is returned as a numpy array with shape (3, 3).

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`rotation_matrix` to get or set the rotation matrix in the convention of the frame.
            - method :meth:`set_rotation_matrix` to set the rotation matrix in a specific convention.
            - method :meth:`get_global_rotation_matrix` to get the rotation matrix between the global frame and this frame.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        Returns
        -------
        numpy.ndarray
            The rotation matrix between the parent frame and this frame in the given convention with shape (3, 3).

        Examples
        --------
        Lets create a frame from its axes and origin with the default convention 0.

        .. code-block:: python

            import numpy
            from py3dframe import Frame

            origin = numpy.array([-1, -2, -3])
            x_axis = numpy.array([1, 1, 0]) / numpy.sqrt(2)
            y_axis = numpy.array([-1, 1, 0]) / numpy.sqrt(2)
            z_axis = numpy.array([0, 0, 1])

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.from_axes(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, convention=0, parent=parent)

        If an application using an other convention required the rotation matrix in the convention 4, you can access the rotation matrix in this convention with the :meth:`get_rotation_matrix` method.

        .. code-block:: python

            rotation_matrix_convention_4 = frame.get_rotation_matrix(convention=4)

        """
        return self.get_rotation(convention=convention).as_matrix()
    
    def set_rotation_matrix(self, rotation_matrix: numpy.ndarray, *, convention: Optional[int] = None) -> None:
        r"""
        Set the rotation matrix between the parent frame and this frame in a specific convention.

        The rotation matrix must be given as a array-like with shape (3, 3).

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`rotation_matrix` to get or set the rotation matrix in the convention of the frame.
            - method :meth:`get_rotation_matrix` to get the rotation matrix in a specific convention.
            - method :meth:`set_global_rotation_matrix` to set the rotation matrix between the global frame and this frame.

        Parameters
        ----------
        rotation_matrix : numpy.ndarray
            The rotation matrix between the parent frame and this frame in the given convention as an array-like with shape (3, 3).

        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        Examples
        --------
        Lets create a default frame with convention 0.

        .. code-block:: python

            from py3dframe import Frame

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.canonical(convention=0, parent=parent)

        Lets assume, an application uses convention 4 to represent the transformation between two frames.
        The frame of reference of the application rotates and the new rotation matrix between the parent frame and this frame in convention 4 can be extracted from the application.

        .. code-block:: python

            import numpy

            rotation_matrix = numpy.array(...) # Extract the rotation matrix from the application.

            frame.set_rotation_matrix(rotation_matrix, convention=4) # Set the rotation matrix in convention 4.

        """
        rotation_matrix = self._format_rotation_matrix_dev(rotation_matrix)
        R = Rotation.from_matrix(rotation_matrix)
        self.set_rotation(R, convention=convention)

    @property
    def rotation_matrix(self) -> numpy.ndarray:
        r"""
        The rotation matrix between the parent frame and this frame in the convention of the frame.

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - method :meth:`get_rotation_matrix` to get the rotation matrix in a specific convention.
            - method :meth:`set_rotation_matrix` to set the rotation matrix in a specific convention.

        Parameters
        ----------
        rotation_matrix : numpy.ndarray
            The rotation matrix between the parent frame and this frame in the convention of the frame as an array-like with shape (3, 3).

        Returns
        -------
        numpy.ndarray
            The rotation matrix between the parent frame and this frame in the convention of the frame with shape (3, 3).

        """
        return self.get_rotation_matrix(convention=self._convention)

    @rotation_matrix.setter
    def rotation_matrix(self, rotation_matrix: numpy.ndarray) -> None:
        self.set_rotation_matrix(rotation_matrix, convention=self._convention)



    def get_quaternion(self, *, convention: Optional[int] = None, scalar_first: bool = True) -> numpy.ndarray:
        r"""
        Access the quaternion representation of the rotation between the parent frame and this frame in a specific convention.

        The quaternion is returned as a numpy array with shape (4,) in the scalar first [w, x, y, z] or scalar last [x, y, z, w] convention.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`quaternion` to get or set the quaternion in the convention of the frame.
            - method :meth:`set_quaternion` to set the quaternion in a specific convention.
            - method :meth:`get_global_quaternion` to get the quaternion between the global frame and this frame.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        scalar_first : bool, optional
            If True, the quaternion is in the scalar first convention. Default is True. If False, the quaternion is in the scalar last convention.

        Returns
        -------
        numpy.ndarray
            The quaternion between the parent frame and this frame in the given convention with shape (4,).

        Examples
        --------
        Lets create a frame from its axes and origin with the default convention 0.

        .. code-block:: python

            import numpy
            from py3dframe import Frame

            origin = numpy.array([-1, -2, -3])
            x_axis = numpy.array([1, 1, 0]) / numpy.sqrt(2)
            y_axis = numpy.array([-1, 1, 0]) / numpy.sqrt(2)
            z_axis = numpy.array([0, 0, 1])

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.from_axes(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, convention=0, parent=parent)

        If an application using an other convention required the quaternion in the convention 4 with scalar first convention, you can access the quaternion in this convention with the :meth:`get_quaternion` method.

        .. code-block:: python

            quaternion_convention_4 = frame.get_quaternion(convention=4, scalar_first=True)

        """
        if not isinstance(scalar_first, bool):
            raise TypeError("The scalar_first parameter must be a boolean.")
        return self.get_rotation(convention=convention).as_quat(scalar_first=scalar_first)
    
    def set_quaternion(self, quaternion: numpy.ndarray, *, convention: Optional[int] = None, scalar_first: bool = True) -> None:
        r"""
        Set the quaternion representation of the rotation between the parent frame and this frame in a specific convention.

        The quaternion must be given as a array-like with 4 elements in the scalar first [w, x, y, z] or scalar last [x, y, z, w] convention.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`quaternion` to get or set the quaternion in the convention of the frame.
            - method :meth:`get_quaternion` to get the quaternion in a specific convention.
            - method :meth:`set_global_quaternion` to set the quaternion between the global frame and this frame.

        Parameters
        ----------
        quaternion : numpy.ndarray
            The quaternion between the parent frame and this frame in the given convention as an array-like with 4 elements.

        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        scalar_first : bool, optional
            If True, the quaternion is in the scalar first convention. Default is True. If False, the quaternion is in the scalar last convention.

        Examples
        --------
        Lets create a default frame with convention 0.

        .. code-block:: python

            from py3dframe import Frame

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.canonical(convention=0, parent=parent)

        Lets assume, an application uses convention 4 to represent the transformation between two frames.
        The frame of reference of the application rotates and the new quaternion between the parent frame and this frame in convention 4 with scalar first convention can be extracted from the application.

        .. code-block:: python

            import numpy

            quaternion = numpy.array(...) # Extract the quaternion from the application.

            frame.set_quaternion(quaternion, convention=4, scalar_first=True) # Set the quaternion in convention 4 with scalar first convention.

        """
        if not isinstance(scalar_first, bool):
            raise TypeError("The scalar_first parameter must be a boolean.")
        quaternion = self._format_quaternion_dev(quaternion)
        R = Rotation.from_quat(quaternion, scalar_first=scalar_first)
        self.set_rotation(R, convention=convention)
    
    @property
    def quaternion(self) -> numpy.ndarray:
        r"""
        The quaternion representation of the rotation between the parent frame and this frame in the convention of the frame.

        The quaternion is in the scalar first convention [w, x, y, z].

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - method :meth:`get_quaternion` to get the quaternion in a specific convention and order.
            - method :meth:`set_quaternion` to set the quaternion in a specific convention and order.

        Parameters
        ----------
        quaternion : numpy.ndarray
            The quaternion between the parent frame and this frame in the convention of the frame as an array-like with 4 elements. The quaternion is in the scalar first convention [w, x, y, z].

        Returns
        -------
        numpy.ndarray
            The quaternion between the parent frame and this frame in the convention of the frame with shape (4,). The quaternion is in the scalar first convention [w, x, y, z].
       
        """
        return self.get_quaternion(convention=self._convention, scalar_first=True)
    
    @quaternion.setter
    def quaternion(self, quaternion: numpy.ndarray) -> None:
        self.set_quaternion(quaternion, convention=self._convention, scalar_first=True)

    
    def get_euler_angles(self, *, convention: Optional[int] = None, degrees: bool = False, seq: str = "xyz") -> numpy.ndarray:
        r"""
        Access the Euler angles representation of the rotation between the parent frame and this frame in a specific convention.

        The Euler angles describe the rotation using three elementary rotations about specified axes. 

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`euler_angles` to get or set the Euler angles in the convention of the frame.
            - method :meth:`set_euler_angles` to set the Euler angles in a specific convention.
            - method :meth:`get_global_euler_angles` to get the Euler angles between the global frame and this frame.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        degrees : bool, optional
            If True, the Euler angles are returned in degrees. Default is False (radians).

        seq : str, optional
            The axes of the Euler angles. Default is "xyz". It must be a string of 3 characters chosen among 'X', 'Y', 'Z', 'x', 'y', 'z'.

        Returns
        -------
        numpy.ndarray
            The Euler angles between the parent frame and this frame in the given convention with shape (3,).

        Examples
        --------
        Lets create a frame from its axes and origin with the default convention 0.

        .. code-block:: python

            import numpy
            from py3dframe import Frame

            origin = numpy.array([-1, -2, -3])
            x_axis = numpy.array([1, 1, 0]) / numpy.sqrt(2)
            y_axis = numpy.array([-1, 1, 0]) / numpy.sqrt(2)
            z_axis = numpy.array([0, 0, 1])

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.from_axes(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, convention=0, parent=parent)

        If an application using an other convention required the Euler angles in the convention 4 with 'ZYX' sequence in degrees, you can access the Euler angles in this convention with the :meth:`get_euler_angles` method.

        .. code-block:: python

            euler_angles_convention_4 = frame.get_euler_angles(convention=4, degrees=True, seq='ZYX')

        """
        if not isinstance(degrees, bool):
            raise TypeError("The degrees parameter must be a boolean.")
        if not isinstance(seq, str):
            raise TypeError("The seq parameter must be a string.")
        if not len(seq) == 3:
            raise ValueError("The seq parameter must have 3 characters.")
        if not all([s in 'XYZxyz' for s in seq]):
            raise ValueError("The seq must contain only the characters 'X', 'Y', 'Z', 'x', 'y', 'z'.") 
        return self.get_rotation(convention=convention).as_euler(seq, degrees=degrees)
    
    def set_euler_angles(self, euler_angles: numpy.ndarray, *, convention: Optional[int] = None, degrees: bool = False, seq: str = "xyz") -> None:
        r"""
        Set the Euler angles representation of the rotation between the parent frame and this frame in a specific convention.

        The Euler angles describe the rotation using three elementary rotations about specified axes.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`euler_angles` to get or set the Euler angles in the convention of the frame.
            - method :meth:`get_euler_angles` to get the Euler angles in a specific convention.
            - method :meth:`set_global_euler_angles` to set the Euler angles between the global frame and this frame.

        Parameters
        ----------
        euler_angles : numpy.ndarray
            The Euler angles between the parent frame and this frame in the given convention as an array-like with 3 elements.

        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        degrees : bool, optional
            If True, the Euler angles are in degrees. Default is False (radians).

        seq : str, optional
            The axes of the Euler angles. Default is "xyz". It must be a string of 3 characters chosen among 'X', 'Y', 'Z', 'x', 'y', 'z'.

        Examples
        --------
        Lets create a default frame with convention 0.

        .. code-block:: python

            from py3dframe import Frame

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.canonical(convention=0, parent=parent)
        
        Lets assume, an application uses convention 4 to represent the transformation between two frames.
        The frame of reference of the application rotates and the new Euler angles between the parent frame and this frame in convention 4 with 'ZYX' sequence in degrees can be extracted from the application.

        .. code-block:: python

            import numpy

            euler_angles = numpy.array(...) # Extract the Euler angles from the application.

            frame.set_euler_angles(euler_angles, convention=4, degrees=True, seq='ZYX') # Set the Euler angles in convention 4 with 'ZYX' sequence in degrees.
        
        """
        if not isinstance(degrees, bool):
            raise TypeError("The degrees parameter must be a boolean.")
        if not isinstance(seq, str):
            raise TypeError("The seq parameter must be a string.")
        if not len(seq) == 3:
            raise ValueError("The seq parameter must have 3 characters.")
        if not all([s in 'XYZxyz' for s in seq]):
            raise ValueError("The seq must contain only the characters 'X', 'Y', 'Z', 'x', 'y', 'z'.") 
        euler_angles = self._format_euler_angles_dev(euler_angles)
        R = Rotation.from_euler(seq, euler_angles, degrees=degrees)
        self.set_rotation(R, convention=convention)

    @property
    def euler_angles(self) -> numpy.ndarray:
        r"""
        The Euler angles representation of the rotation between the parent frame and this frame in the convention of the frame.

        The Euler angles are in radians and the sequence is 'xyz'.

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - method :meth:`get_euler_angles` to get the Euler angles in a specific convention, sequence and unit.
            - method :meth:`set_euler_angles` to set the Euler angles in a specific convention, sequence and unit.

        Parameters
        ----------
        euler_angles : numpy.ndarray
            The Euler angles between the parent frame and this frame in the convention of the frame with shape (3,). The angles are in radians and the sequence is 'xyz'.

        Returns
        -------
        numpy.ndarray
            The Euler angles between the parent frame and this frame in the convention of the frame with shape (3,). The angles are in radians and the sequence is 'xyz'.

        """
        return self.get_euler_angles(convention=self._convention, degrees=False, seq="xyz")

    @euler_angles.setter
    def euler_angles(self, euler_angles: numpy.ndarray) -> None:
        self.set_euler_angles(euler_angles, convention=self._convention, degrees=False, seq="xyz")



    def get_rotation_vector(self, *, convention: Optional[int] = None, degrees: bool = False) -> numpy.ndarray:
        r"""
        Access the rotation vector representation of the rotation between the parent frame and this frame in a specific convention.

        The rotation vector describes the rotation by a single rotation about a fixed axis. The direction of the axis is given by the direction of the vector, and the magnitude of the vector is given by the angle of rotation.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`rotation_vector` to get or set the rotation vector in the convention of the frame.
            - method :meth:`set_rotation_vector` to set the rotation vector in a specific convention.
            - method :meth:`get_global_rotation_vector` to get the rotation vector between the global frame and this frame.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        degrees : bool, optional
            If True, the rotation vector is returned in degrees. Default is False (radians).

        Returns
        -------
        numpy.ndarray
            The rotation vector between the parent frame and this frame in the given convention with shape (3,).

        Examples
        --------
        Lets create a frame from its axes and origin with the default convention 0.

        .. code-block:: python

            import numpy
            from py3dframe import Frame

            origin = numpy.array([-1, -2, -3])
            x_axis = numpy.array([1, 1, 0]) / numpy.sqrt(2)
            y_axis = numpy.array([-1, 1, 0]) / numpy.sqrt(2)
            z_axis = numpy.array([0, 0, 1])

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.from_axes(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, convention=0, parent=parent)

        If an application using an other convention required the rotation vector in the convention 4 in degrees, you can access the rotation vector in this convention with the :meth:`get_rotation_vector` method.
        
        .. code-block:: python

            rotation_vector_convention_4 = frame.get_rotation_vector(convention=4, degrees=True)

        """
        if not isinstance(degrees, bool):
            raise TypeError("The degrees parameter must be a boolean.")
        return self.get_rotation(convention=convention).as_rotvec(degrees=degrees)

    def set_rotation_vector(self, rotation_vector: numpy.ndarray, *, convention: Optional[int] = None, degrees: bool = False) -> None:
        r"""
        Set the rotation vector representation of the rotation between the parent frame and this frame in a specific convention.

        The rotation vector describes the rotation by a single rotation about a fixed axis. The direction of the axis is given by the direction of the vector, and the magnitude of the vector is given by the angle of rotation.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`rotation_vector` to get or set the rotation vector in the convention of the frame.
            - method :meth:`get_rotation_vector` to get the rotation vector in a specific convention.
            - method :meth:`set_global_rotation_vector` to set the rotation vector between the global frame and this frame.

        Parameters
        ----------
        rotation_vector : numpy.ndarray
            The rotation vector between the parent frame and this frame in the given convention as an array-like with 3 elements.

        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        degrees : bool, optional
            If True, the rotation vector is in degrees. Default is False (radians).

        Examples
        --------
        Lets create a default frame with convention 0.

        .. code-block:: python

            from py3dframe import Frame

            parent = ... # Define the parent frame if needed, otherwise parent=None to use the canonical frame.

            frame = Frame.canonical(convention=0, parent=parent)

        Lets assume, an application uses convention 4 to represent the transformation between two frames.
        The frame of reference of the application rotates and the new rotation vector between the parent frame and this frame in convention 4 in degrees can be extracted from the application.

        .. code-block:: python

            import numpy

            rotation_vector = numpy.array(...) # Extract the rotation vector from the application.

            frame.set_rotation_vector(rotation_vector, convention=4, degrees=True) # Set the rotation vector in convention 4 in degrees.

        """
        if not isinstance(degrees, bool):
            raise TypeError("The degrees parameter must be a boolean.")
        rotation_vector = self._format_rotation_vector_dev(rotation_vector)
        R = Rotation.from_rotvec(rotation_vector, degrees=degrees)
        self.set_rotation(R, convention=convention)
    
    @property
    def rotation_vector(self) -> numpy.ndarray:
        r"""
        The rotation vector representation of the rotation between the parent frame and this frame in the convention of the frame.

        The rotation vector is in radians.

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - method :meth:`get_rotation_vector` to get the rotation vector in a specific convention and unit.
            - method :meth:`set_rotation_vector` to set the rotation vector in a specific convention and unit.

        Parameters
        ----------
        rotation_vector : numpy.ndarray
            The rotation vector between the parent frame and this frame in the convention of the frame with shape (3,). The rotation vector is in radians.

        Returns
        -------
        numpy.ndarray
            The rotation vector between the parent frame and this frame in the convention of the frame with shape (3,). The rotation vector is in radians.

        """
        return self.get_rotation_vector(convention=self._convention, degrees=False)
    
    @rotation_vector.setter
    def rotation_vector(self, rotation_vector: numpy.ndarray) -> None:
        self.set_rotation_vector(rotation_vector, convention=self._convention, degrees=False)



    # ====================================================================================================================================
    # Global transformation methods
    # ====================================================================================================================================
    def get_global_frame(self) -> Frame:
        r"""
        Get the global frame of the frame. The representation of the frame with respect to the global canonical frame.

        If the frame has no parent, the global frame is the frame itself.

        Otherwise, the global frame is computed by composing the transformation between the global frame and the parent frame with the transformation between the parent frame and this frame.

        .. note::

            The parent attribute of the global frame is None.

        .. warning::

            The Frame object is a new object. Any change in the returned object will not affect the original object.
            Furthermore, any change in the original object will not affect the returned object.
            It describes the frame at the state of the call.

        Returns
        -------
        Frame
            The global frame of the frame.

        Examples
        --------
        Lets consider a train and a person standing into the train.

        First we can define the train frame relative to the global frame (the canonical frame) and then we can define the person frame relative to the train frame.

        .. code-block:: python

            from py3dframe import Frame

            train = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=None)
            person = Frame.from_axes(origin=[0, 1, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=train)

        Lets assume the train moves and the person moves inside the train.

        .. code-block:: python

            train.set_translation([10, 0, 0], convention=0) # The train moves of 10 units along the x axis of the global frame.

            person.set_translation([0, 2, 0], convention=0) # The person moves by 1 unit along the y axis of the train frame.
            person.set_euler_angles([0, 0, 45], convention=0, degrees=True) # The person rotates of 45 degrees around the z axis of the train frame.

        We can determine the orientation and position of the person relative to the global frame at any time using the :meth:`get_global_frame` method:

        .. code-block:: python

            global_person = person.get_global_frame() # Get the global frame of the person.

            print("Person origin in the global frame:", global_person.origin)
            print("Person x axis in the global frame:", global_person.x_axis)
            print("Person y axis in the global frame:", global_person.y_axis)
            print("Person z axis in the global frame:", global_person.z_axis)
            # Output:
            # Person origin in the global frame: [[10] [2] [0]]
            # Person x axis in the global frame: [[ 0.70710678] [ 0.70710678] [ 0.        ]]
            # Person y axis in the global frame: [[-0.70710678] [ 0.70710678] [0.        ]]
            # Person z axis in the global frame: [[0.] [0.] [1.]]

        """
        if self._parent is None:
            global_frame = self
        
        else:
            # Construct the composite transformation between the global frame and this frame.
            R_parent = self._parent.get_global_rotation(convention=0)
            T_parent = self._parent.get_global_translation(convention=0)

            # ====================================================================================================================================
            # Lets note : 
            # Xg : the coordinates of a point in the global frame
            # Xp : the coordinates of the same point in the parent frame
            # Xf : the coordinates of the same point in the frame
            # Rg : the rotation matrix between the global frame and the parent frame
            # Rp : the rotation matrix between the parent frame and this frame
            # Tg : the translation vector between the global frame and the parent frame
            # Tp : the translation vector between the parent frame and this frame
            # R : the rotation matrix between the global frame and this frame
            # T : the translation vector between the global frame and this frame
            # 
            # We have :
            # Xg = Rg * Xp + Tg
            # Xp = Rp * Xf + Tp
            # Xg = R * Xf + T
            #
            # We search R and T:
            # Xg = Rg * (Rp * Xf + Tp) + Tg
            # Xg = Rg * Rp * Xf + Rg * Tp + Tg
            #
            # So the composite transformation is R = Rg * Rp and T = Rg * Tp + Tg
            # ====================================================================================================================================
            rotation = R_parent * self._R_dev
            translation = R_parent.apply(self._T_dev.T).T + T_parent
            global_frame = Frame.from_rotation(translation=translation, rotation=rotation, parent=None, convention=0)
            global_frame.convention = self._convention

        return global_frame.copy()
    

    def get_global_rotation(self, *, convention: Optional[int] = None) -> scipy.spatial.transform.Rotation:
        r"""
        Access the rotation between the global frame and this frame in the given convention.

        The rotation is returned as a :class:`Rotation` object.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`global_rotation` to get or set the rotation between the global frame and this frame in the convention of the frame.
            - method :meth:`set_global_rotation` to set the rotation between the global frame and this frame in a specific convention.
            - method :meth:`get_rotation` to get the rotation between the parent frame and this frame in a specific convention.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        Returns
        -------
        Rotation
            The rotation between the global frame and this frame in the given convention.

        Examples
        --------
        Lets consider a train and a person standing into the train.

        First we can define the train frame relative to the global frame (the canonical frame) and then we can define the person frame relative to the train frame.

        .. code-block:: python

            from py3dframe import Frame

            train = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=None)
            person = Frame.from_axes(origin=[0, 1, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=train)

        Lets assume the train moves and the person moves inside the train.

        .. code-block:: python

            train.set_translation([10, 0, 0], convention=0) # The train moves of 10 units along the x axis of the global frame.

            person.set_translation([0, 2, 0], convention=0) # The person moves by 1 unit along the y axis of the train frame.
            person.set_euler_angles([0, 0, 45], convention=0, degrees=True) # The person rotates of 45 degrees around the z axis of the train frame.

        If an application using an other convention required the rotation between the global frame and the person frame in the convention 4, you can access this rotation with the :meth:`get_global_rotation` method:

        .. code-block:: python

            global_rotation_person = person.get_global_rotation(convention=4) # Get the rotation between the global frame and the person frame in convention 4.

        """
        global_frame = self.get_global_frame()
        return global_frame.get_rotation(convention=convention)


    def set_global_rotation(self, rotation: scipy.spatial.transform.Rotation, *, convention: Optional[int] = None) -> None:
        r"""
        Set the rotation between the global frame and this frame in the given convention.

        The rotation must be given as a :class:`Rotation` object.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`global_rotation` to get or set the rotation between the global frame and this frame in the convention of the frame.
            - method :meth:`get_global_rotation` to get the rotation between the global frame and this frame in a specific convention.
            - method :meth:`set_rotation` to set the rotation between the parent frame and this frame in a specific convention.

        Parameters
        ----------
        rotation : Rotation
            The rotation between the global frame and this frame in the given convention.

        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        Examples
        --------
        Lets consider a train and a person standing into the train.

        First we can define the train frame relative to the global frame (the canonical frame) and then we can define the person frame relative to the train frame.

        .. code-block:: python

            from py3dframe import Frame

            train = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=None)
            person = Frame.from_axes(origin=[0, 1, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=train)
        
        Lets assume the train moves.

        .. code-block:: python

            train.set_translation([10, 0, 0], convention=0) # The train moves of 10 units along the x axis of the global frame.

        If an application using an other convention gives the new rotation between the global frame and the person frame in the convention 4, you can set this rotation with the :meth:`set_global_rotation` method:

        .. code-block:: python

            import numpy

            rotation = ... # Get the new rotation between the global frame and the person frame in convention 4 from the application.

            person.set_global_rotation(rotation, convention=4) # Set the rotation between the global frame and the person frame in convention 4.

        The person frame is updated accordingly to keep the correct transformation between the global frame and the person frame.

        Then we can determine the new orientation and position of the person relative to the train frame:

        .. code-block:: python

            print("Person origin in the train frame:", person.origin)
            print("Person axes in the train frame:", person.axes)

        """
        if self._parent is None:
            self.set_rotation(rotation, convention=convention)
            return
        
        # ====================================================================================================================================
        # Lets note : 
        # Xg : the coordinates of a point in the global frame
        # Xp : the coordinates of the same point in the parent frame
        # Xf : the coordinates of the same point in the frame
        # Rg : the rotation matrix between the global frame and the parent frame
        # Rp : the rotation matrix between the parent frame and this frame
        # Tg : the translation vector between the global frame and the parent frame
        # Tp : the translation vector between the parent frame and this frame
        # R : the rotation matrix between the global frame and this frame
        # T : the translation vector between the global frame and this frame
        # 
        # We have :
        # Xg = Rg * Xp + Tg
        # Xp = Rp * Xf + Tp
        # Xg = R * Xf + T
        #
        # We search Rp:
        # Xg = Rg * (Rp * Xf + Tp) + Tg
        # Xg = Rg * Rp * Xf + Rg * Tp + Tg
        # R = Rg * Rp
        # T = Rg * Tp + Tg
        #
        # So Rp = Rg.inv() * R
        # ====================================================================================================================================

        convention = self._format_convention_dev(convention, allow_None=True)
        R_parent = self._parent.get_global_rotation(convention=0)

        rotation, _ = switch_RT_convention(rotation, self._T_dev, convention, 0)
        rotation = R_parent.inv() * rotation
        self.set_rotation(rotation, convention=0)

    @property
    def global_rotation(self) -> scipy.spatial.transform.Rotation:
        r"""
        The rotation between the global frame and this frame in the convention of the frame.

        .. note::

            This property is settable.

        .. seealso::  

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - method :meth:`get_global_rotation` to get the rotation in a specific convention.
            - method :meth:`set_global_rotation` to set the rotation in a specific convention.

        Parameters
        ----------
        rotation : Rotation
            The rotation between the global frame and this frame in the convention of the frame.

        Returns
        -------
        Rotation
            The rotation between the global frame and this frame in the convention of the frame.

        """
        return self.get_global_rotation(convention=self._convention)

    @global_rotation.setter
    def global_rotation(self, rotation: scipy.spatial.transform.Rotation) -> None:
        self.set_global_rotation(rotation, convention=self._convention)
    


    def get_global_translation(self, *, convention: Optional[int] = None) -> numpy.ndarray:
        r"""
        Access the translation vector between the global frame and this frame in the given convention.

        The translation vector has shape (3, 1).

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`global_translation` to get or set the translation vector between the global frame and this frame in the convention of the frame.
            - method :meth:`set_global_translation` to set the translation vector between the global frame and this frame in a specific convention.
            - method :meth:`get_translation` to get the translation vector between the parent frame and this frame in a specific convention.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        Returns
        -------
        numpy.ndarray
            The translation vector between the global frame and this frame in the given convention with shape (3, 1).

        Examples
        --------
        Lets consider a train and a person standing into the train.

        First we can define the train frame relative to the global frame (the canonical frame) and then we can define the person frame relative to the train frame.

        .. code-block:: python

            from py3dframe import Frame

            train = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=None)
            person = Frame.from_axes(origin=[0, 1, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=train)
        
        Lets assume the train moves and the person moves inside the train.

        .. code-block:: python

            train.set_translation([10, 0, 0], convention=0) # The train moves of 10 units along the x axis of the global frame.

            person.set_translation([0, 2, 0], convention=0) # The person moves by 1 unit along the y axis of the train frame.
            person.set_euler_angles([0, 0, 45], convention=0, degrees=True) # The person rotates of 45 degrees around the z axis of the train frame.

        If an application using an other convention required the translation between the global frame and the person frame in the convention 4, you can access this translation with the :meth:`get_global_translation` method:

        .. code-block:: python

            global_translation_person = person.get_global_translation(convention=4) # Get the translation between the global frame and the person frame in convention 4.

        """
        global_frame = self.get_global_frame()
        return global_frame.get_translation(convention=convention)
    
    def set_global_translation(self, translation: numpy.ndarray, *, convention: Optional[int] = None) -> None:
        r"""
        Set the translation vector between the global frame and this frame in the given convention.

        The translation vector must be a array-like with 3 elements.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`global_translation` to get or set the translation vector between the global frame and this frame in the convention of the frame.
            - method :meth:`get_global_translation` to get the translation vector between the global frame and this frame in a specific convention.
            - method :meth:`set_translation` to set the translation vector between the parent frame and this frame in a specific convention.

        Parameters
        ----------
        translation : numpy.ndarray
            The translation vector between the global frame and this frame in the given convention as an array-like with 3 elements.

        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        Examples
        --------
        Lets consider a train and a person standing into the train.

        First we can define the train frame relative to the global frame (the canonical frame) and then we can define the person frame relative to the train frame.

        .. code-block:: python

            from py3dframe import Frame

            train = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=None)
            person = Frame.from_axes(origin=[0, 1, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=train)
        
        Lets assume the train moves.

        .. code-block:: python

            train.set_translation([10, 0, 0], convention=0) # The train moves of 10 units along the x axis of the global frame.

        If an application using an other convention gives the new translation between the global frame and the person frame in the convention 4, you can set this translation with the :meth:`set_global_translation` method:

        .. code-block:: python

            import numpy

            translation = ... # Get the new translation between the global frame and the person frame in convention 4 from the application.

            person.set_global_translation(translation, convention=4) # Set the translation between the global frame and the person frame in convention 4.

        The person frame is updated accordingly to keep the correct transformation between the global frame and the person frame.

        Then we can determine the new orientation and position of the person relative to the train frame:

        .. code-block:: python

            print("Person origin in the train frame:", person.origin)
            print("Person axes in the train frame:", person.axes)

        """
        if self._parent is None:
            self.set_translation(translation, convention=convention)
            return
        
        # ====================================================================================================================================
        # Lets note : 
        # Xg : the coordinates of a point in the global frame
        # Xp : the coordinates of the same point in the parent frame
        # Xf : the coordinates of the same point in the frame
        # Rg : the rotation matrix between the global frame and the parent frame
        # Rp : the rotation matrix between the parent frame and this frame
        # Tg : the translation vector between the global frame and the parent frame
        # Tp : the translation vector between the parent frame and this frame
        # R : the rotation matrix between the global frame and this frame
        # T : the translation vector between the global frame and this frame
        # 
        # We have :
        # Xg = Rg * Xp + Tg
        # Xp = Rp * Xf + Tp
        # Xg = R * Xf + T
        #
        # We search Tp:
        # Xg = Rg * (Rp * Xf + Tp) + Tg
        # Xg = Rg * Rp * Xf + Rg * Tp + Tg
        # R = Rg * Rp
        # T = Rg * Tp + Tg
        #
        # So Tp = Rg.inv() * (T - Tg)
        # ====================================================================================================================================

        convention = self._format_convention_dev(convention, allow_None=True)
        R_parent = self._parent.get_global_rotation(convention=0)
        T_parent = self._parent.get_global_translation(convention=0)

        _, translation = switch_RT_convention(self._R_dev, translation, 0, convention)
        translation = R_parent.inv().apply((translation - T_parent).T).T
        self.set_translation(translation, convention=0)

    @property
    def global_translation(self) -> numpy.ndarray:
        r"""
        The translation vector between the global frame and this frame in the convention of the frame.

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - method :meth:`get_global_translation` to get the translation vector in a specific convention.
            - method :meth:`set_global_translation` to set the translation vector in a specific convention.

        Parameters
        ----------
        translation : numpy.ndarray
            The translation vector between the global frame and this frame in the convention of the frame as an array-like with 3 elements.

        Returns
        -------
        numpy.ndarray
            The translation vector between the global frame and this frame in the convention of the frame with shape (3, 1).

        """
        return self.get_global_translation(convention=self._convention)

    @global_translation.setter
    def global_translation(self, translation: numpy.ndarray) -> None:
        self.set_global_translation(translation, convention=self._convention)
    

    def get_global_rotation_matrix(self, *, convention: Optional[int] = None) -> numpy.ndarray:
        r"""
        Access the rotation matrix representation of the rotation between the global frame and this frame in the given convention.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        Returns
        -------
        numpy.ndarray
            The rotation matrix between the global frame and this frame in the given convention with shape (3, 3).

        Examples
        --------
        Lets consider a train and a person standing into the train.

        First we can define the train frame relative to the global frame (the canonical frame) and then we can define the person frame relative to the train frame.

        .. code-block:: python

            from py3dframe import Frame

            train = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=None)
            person = Frame.from_axes(origin=[0, 1, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=train)
        
        Lets assume the train moves and the person moves inside the train.

        .. code-block:: python

            train.set_translation([10, 0, 0], convention=0) # The train moves of 10 units along the x axis of the global frame.

            person.set_translation([0, 2, 0], convention=0) # The person moves by 1 unit along the y axis of the train frame.
            person.set_euler_angles([0, 0, 45], convention=0, degrees=True) # The person rotates of 45 degrees around the z axis of the train frame.

        If an application using an other convention required the rotation matrix between the global frame and the person frame in the convention 4, you can access this rotation matrix with the :meth:`get_global_rotation_matrix` method:

        .. code-block:: python

            global_rotation_person = person.get_global_rotation_matrix(convention=4) # Get the rotation matrix between the global frame and the person frame in convention 4.

        """
        global_frame = self.get_global_frame()
        return global_frame.get_rotation_matrix(convention=convention)

    def set_global_rotation_matrix(self, rotation_matrix: numpy.ndarray, *, convention: Optional[int] = None) -> None:
        r"""
        Set the rotation matrix representation of the rotation between the global frame and this frame in the given convention.

        Parameters
        ----------
        rotation_matrix : numpy.ndarray
            The rotation matrix between the global frame and this frame in the given convention with shape (3, 3).

        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        Examples
        --------
        Lets consider a train and a person standing into the train.

        First we can define the train frame relative to the global frame (the canonical frame) and then we can define the person frame relative to the train frame.

        .. code-block:: python

            from py3dframe import Frame

            train = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=None)
            person = Frame.from_axes(origin=[0, 1, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=train)
        
        Lets assume the train moves.

        .. code-block:: python

            train.set_translation([10, 0, 0], convention=0) # The train moves of 10 units along the x axis of the global frame.

        If an application using an other convention gives the new rotation matrix between the global frame and the person frame in the convention 4, you can set this rotation with the :meth:`set_global_rotation_matrix` method:

        .. code-block:: python

            import numpy

            rotation_matrix = ... # Get the new rotation matrix between the global frame and the person frame in convention 4 from the application.

            person.set_global_rotation_matrix(rotation_matrix, convention=4) # Set the rotation matrix between the global frame and the person frame in convention 4.

        The person frame is updated accordingly to keep the correct transformation between the global frame and the person frame.

        Then we can determine the new orientation and position of the person relative to the train frame:

        .. code-block:: python

            print("Person origin in the train frame:", person.origin)
            print("Person axes in the train frame:", person.axes)

        """
        rotation_matrix = self._format_rotation_matrix_dev(rotation_matrix)
        R = Rotation.from_matrix(rotation_matrix)
        self.set_global_rotation(R, convention=convention)
    
    @property
    def global_rotation_matrix(self) -> numpy.ndarray:
        r"""
        The rotation matrix representation of the rotation between the global frame and this frame in the convention of the frame.

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - method :meth:`get_global_rotation_matrix` to get the rotation matrix in a specific convention.
            - method :meth:`set_global_rotation_matrix` to set the rotation matrix in a specific convention.

        Parameters
        ----------
        rotation_matrix : numpy.ndarray
            The rotation matrix between the global frame and this frame in the convention of the frame with shape (3, 3).

        Returns
        -------
        numpy.ndarray
            The rotation matrix between the global frame and this frame in the convention of the frame with shape (3, 3).

        """
        return self.get_global_rotation_matrix(convention=self._convention)

    @global_rotation_matrix.setter
    def global_rotation_matrix(self, rotation_matrix: numpy.ndarray) -> None:
        self.set_global_rotation_matrix(rotation_matrix, convention=self._convention)


    
    def get_global_quaternion(self, *, convention: Optional[int] = None, scalar_first: bool = True) -> numpy.ndarray:
        r"""
        Access the quaternion representation of the rotation between the global frame and this frame in the given convention.

        The quaternion is returned as a numpy array with shape (4,) in the scalar first [w, x, y, z] or scalar last [x, y, z, w] convention.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`global_quaternion` to get or set the quaternion between the global frame and this frame in the convention of the frame.
            - method :meth:`set_global_quaternion` to set the quaternion between the global frame and this frame in a specific convention.
            - method :meth:`get_quaternion` to get the quaternion between the parent frame and this frame in a specific convention.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        scalar_first : bool, optional
            If True, the quaternion is in the scalar first convention. Default is True.

        Returns
        -------
        numpy.ndarray
            The quaternion between the global frame and this frame in the given convention with shape (4,).

        Examples
        --------
        Lets consider a train and a person standing into the train.

        First we can define the train frame relative to the global frame (the canonical frame) and then we can define the person frame relative to the train frame.

        .. code-block:: python

            from py3dframe import Frame

            train = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=None)
            person = Frame.from_axes(origin=[0, 1, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=train)

        Lets assume the train moves and the person moves inside the train.

        .. code-block:: python

            train.set_translation([10, 0, 0], convention=0) # The train moves of 10 units along the x axis of the global frame.

            person.set_translation([0, 2, 0], convention=0) # The person moves by 1 unit along the y axis of the train frame.
            person.set_euler_angles([0, 0, 45], convention=0, degrees=True) # The person rotates of 45 degrees around the z axis of the train frame.

        If an application using an other convention required the quaternion between the global frame and the person frame in the convention 4 and scalar first convention, you can access this quaternion with the :meth:`get_global_quaternion` method:

        .. code-block:: python

            global_quaternion_person = person.get_global_quaternion(convention=4, scalar_first=True) # Get the quaternion between the global frame and the person frame in convention 4 in the scalar first convention.

        """
        global_frame = self.get_global_frame()
        return global_frame.get_quaternion(convention=convention, scalar_first=scalar_first)

    def set_global_quaternion(self, quaternion: numpy.ndarray, *, convention: Optional[int] = None, scalar_first: bool = True) -> None:
        r"""
        Set the quaternion representation of the rotation between the global frame and this frame in the given convention.

        The quaternion must be a array-like with 4 elements in the scalar first [w, x, y, z] or scalar last [x, y, z, w] convention.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`global_quaternion` to get or set the quaternion between the global frame and this frame in the convention of the frame.
            - method :meth:`get_global_quaternion` to get the quaternion between the global frame and this frame in a specific convention.
            - method :meth:`set_quaternion` to set the quaternion between the parent frame and this frame in a specific convention.

        Parameters
        ----------
        quaternion : numpy.ndarray
            The quaternion between the global frame and this frame in the given convention as an array-like with 4 elements.

        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        scalar_first : bool, optional
            If True, the quaternion is in the scalar first convention. Default is True.

        Examples
        --------
        Lets consider a train and a person standing into the train.

        First we can define the train frame relative to the global frame (the canonical frame) and then we can define the person frame relative to the train frame.

        .. code-block:: python

            from py3dframe import Frame

            train = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=None)
            person = Frame.from_axes(origin=[0, 1, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=train)

        Lets assume the train moves.

        .. code-block:: python

            train.set_translation([10, 0, 0], convention=0) # The train moves of 10 units along the x axis of the global frame.

        If an application using an other convention gives the new quaternion between the global frame and the person frame in the convention 4 and scalar first convention, you can set this quaternion with the :meth:`set_global_quaternion` method:

        .. code-block:: python

            import numpy

            quaternion = ... # Get the new quaternion between the global frame and the person frame in convention 4 in the scalar first convention from the application.

            person.set_global_quaternion(quaternion, convention=4, scalar_first=True) # Set the quaternion between the global frame and the person frame in convention 4 in the scalar first convention.

        The person frame is updated accordingly to keep the correct transformation between the global frame and the person frame.

        Then we can determine the new orientation and position of the person relative to the train frame:

        .. code-block:: python

            print("Person origin in the train frame:", person.origin)
            print("Person axes in the train frame:", person.axes)
        
        """
        quaternion = self._format_quaternion_dev(quaternion)
        R = Rotation.from_quat(quaternion, scalar_first=scalar_first)
        self.set_global_rotation(R, convention=convention)
    
    @property
    def global_quaternion(self) -> numpy.ndarray:
        r"""
        The quaternion representation of the rotation between the global frame and this frame in the convention of the frame.

        The quaternion is in the scalar first convention [w, x, y, z].

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - method :meth:`get_global_quaternion` to get the quaternion in a specific convention.
            - method :meth:`set_global_quaternion` to set the quaternion in a specific convention.

        Parameters
        ----------
        quaternion : numpy.ndarray
            The quaternion between the global frame and this frame in the convention of the frame as an array-like with 4 elements. The quaternion is in the scalar first convention [w, x, y, z].

        Returns
        -------
        numpy.ndarray
            The quaternion between the global frame and this frame in the convention of the frame with shape (4,). The quaternion is in the scalar first convention [w, x, y, z].

        """
        return self.get_global_quaternion(convention=self._convention, scalar_first=True)

    @global_quaternion.setter
    def global_quaternion(self, quaternion: numpy.ndarray) -> None:
        self.set_global_quaternion(quaternion, convention=self._convention, scalar_first=True)
    


    def get_global_euler_angles(self, *, convention: Optional[int] = None, degrees: bool = False, seq: str = "xyz") -> numpy.ndarray:
        r"""
        Access the Euler angles representation of the rotation between the global frame and this frame in the given convention.

        The Euler angles describe the rotation using three elementary rotations about specified axes.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`global_euler_angles` to get or set the Euler angles between the global frame and this frame in the convention of the frame.
            - method :meth:`set_global_euler_angles` to set the Euler angles between the global frame and this frame in a specific convention.
            - method :meth:`get_euler_angles` to get the Euler angles between the parent frame and this frame in a specific convention.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        degrees : bool, optional
            If True, the Euler angles are returned in degrees. Default is False (radians).

        seq : str, optional
            The axes of the Euler angles. It must be a string of 3 characters chosen among 'X', 'Y', 'Z', 'x', 'y', 'z'. Default is "xyz".

        Returns
        -------
        numpy.ndarray
            The Euler angles between the global frame and this frame in the given convention with shape (3,).

        Examples
        --------
        Lets consider a train and a person standing into the train.

        First we can define the train frame relative to the global frame (the canonical frame) and then we can define the person frame relative to the train frame.

        .. code-block:: python

            from py3dframe import Frame

            train = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=None)
            person = Frame.from_axes(origin=[0, 1, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=train)

        Lets assume the train moves and the person moves inside the train.

        .. code-block:: python

            train.set_translation([10, 0, 0], convention=0) # The train moves of 10 units along the x axis of the global frame.
            person.set_translation([0, 2, 0], convention=0) # The person moves by 1 unit along the y axis of the train frame.
            person.set_euler_angles([0, 0, 45], convention=0, degrees=True) # The person rotates of 45 degrees around the z axis of the train frame.

        If an application using an other convention required the Euler angles between the global frame and the person frame in the convention 4, with the axes "zyx" and in degrees, you can access these Euler angles with the :meth:`get_global_euler_angles` method:

        .. code-block:: python

            global_euler_angles_person = person.get_global_euler_angles(convention=4, seq="zyx", degrees=True) # Get the Euler angles between the global frame and the person frame in convention 4 with the axes "zyx" and in degrees.

        """
        global_frame = self.get_global_frame()
        return global_frame.get_euler_angles(convention=convention, degrees=degrees, seq=seq)

    def set_global_euler_angles(self, euler_angles: numpy.ndarray, *, convention: Optional[int] = None, degrees: bool = False, seq: str = "xyz") -> None:
        r"""
        Set the Euler angles representation of the rotation between the global frame and this frame in the given convention.

        The Euler angles must be a array-like with 3 elements representing the angles of rotation about the specified axes.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`global_euler_angles` to get or set the Euler angles between the global frame and this frame in the convention of the frame.
            - method :meth:`get_global_euler_angles` to get the Euler angles between the global frame and this frame in a specific convention.
            - method :meth:`set_euler_angles` to set the Euler angles between the parent frame and this frame in a specific convention.

        Parameters
        ----------
        euler_angles : numpy.ndarray
            The Euler angles between the global frame and this frame in the given convention as an array-like with 3 elements.

        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        degrees : bool, optional
            If True, the Euler angles are in degrees. Default is False (radians).

        seq : str, optional
            The axes of the Euler angles. It must be a string of 3 characters chosen among 'X', 'Y', 'Z', 'x', 'y', 'z'. Default is "xyz".

        Examples
        --------
        Lets consider a train and a person standing into the train.

        First we can define the train frame relative to the global frame (the canonical frame) and then we can define the person frame relative to the train frame.

        .. code-block:: python

            from py3dframe import Frame

            train = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=None)
            person = Frame.from_axes(origin=[0, 1, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=train)

        Lets assume the train moves.

        .. code-block:: python

            train.set_translation([10, 0, 0], convention=0) # The train moves of 10 units along the x axis of the global frame.

        If an application using an other convention gives the new Euler angles between the global frame and the person frame in the convention 4, with the axes "zyx" and in degrees, you can set these Euler angles with the :meth:`set_global_euler_angles` method:

        .. code-block:: python

            import numpy

            euler_angles = ... # Get the new Euler angles between the global frame and the person frame in convention 4 with the axes "zyx" and in degrees from the application.

            person.set_global_euler_angles(euler_angles, convention=4, seq="zyx", degrees=True) # Set the Euler angles between the global frame and the person frame in convention 4 with the axes "zyx" and in degrees.

        The person frame is updated accordingly to keep the correct transformation between the global frame and the person frame.

        Then we can determine the new orientation and position of the person relative to the train frame:

        .. code-block:: python

            print("Person origin in the train frame:", person.origin)
            print("Person axes in the train frame:", person.axes)

        """
        if not isinstance(degrees, bool):
            raise TypeError("The degrees parameter must be a boolean.")
        if not isinstance(seq, str):
            raise TypeError("The seq parameter must be a string.")
        if not len(seq) == 3:
            raise ValueError("The seq parameter must have 3 characters.")
        if not all([s in 'XYZxyz' for s in seq]):
            raise ValueError("The seq must contain only the characters 'X', 'Y', 'Z', 'x', 'y', 'z'.") 
        euler_angles = self._format_euler_angles_dev(euler_angles)
        R = Rotation.from_euler(seq, euler_angles, degrees=degrees)
        self.set_global_rotation(R, convention=convention)

    @property
    def global_euler_angles(self) -> numpy.ndarray:
        r"""
        The Euler angles representation of the rotation between the global frame and this frame in the convention of the frame.

        The Euler angles describe the rotation using three elementary rotations about specified axes in the "xyz" order and are in radians.

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - method :meth:`get_global_euler_angles` to get the Euler angles in a specific convention.
            - method :meth:`set_global_euler_angles` to set the Euler angles in a specific convention.

        Parameters
        ----------
        euler_angles : numpy.ndarray
            The Euler angles between the global frame and this frame in the convention of the frame as an array-like with 3 elements. The Euler angles are in the "xyz" order and in radians.

        Returns
        -------
        numpy.ndarray
            The Euler angles between the global frame and this frame in the convention of the frame with shape (3,). The Euler angles are in the "xyz" order and in radians.

        """
        return self.get_global_euler_angles(convention=self._convention, degrees=False, seq="xyz")

    @global_euler_angles.setter
    def global_euler_angles(self, euler_angles: numpy.ndarray) -> None:
        self.set_global_euler_angles(euler_angles, convention=self._convention, degrees=False, seq="xyz")
    


    def get_global_rotation_vector(self, *, convention: Optional[int] = None, degrees: bool = False) -> numpy.ndarray:
        r"""
        Access the rotation vector representation of the rotation between the global frame and this frame in the given convention.

        The rotation vector is a compact representation of a rotation in 3D space, where the direction of the vector indicates the axis of rotation and the magnitude represents the angle of rotation.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`global_rotation_vector` to get or set the rotation vector between the global frame and this frame in the convention of the frame.
            - method :meth:`set_global_rotation_vector` to set the rotation vector between the global frame and this frame in a specific convention.
            - method :meth:`get_rotation_vector` to get the rotation vector between the parent frame and this frame in a specific convention.

        Parameters
        ----------
        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        degrees : bool, optional
            If True, the rotation vector is returned in degrees. Default is False (radians).

        Returns
        -------
        numpy.ndarray
            The rotation vector between the global frame and this frame in the given convention with shape (3,).

        Examples
        --------
        Lets consider a train and a person standing into the train.

        First we can define the train frame relative to the global frame (the canonical frame) and then we can define the person frame relative to the train frame.

        .. code-block:: python

            from py3dframe import Frame

            train = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=None)
            person = Frame.from_axes(origin=[0, 1, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=train)

        Lets assume the train moves and the person moves inside the train.

        .. code-block:: python

            train.set_translation([10, 0, 0], convention=0) # The train moves of 10 units along the x axis of the global frame.

            person.set_translation([0, 2, 0], convention=0) # The person moves by 1 unit along the y axis of the train frame.
            person.set_euler_angles([0, 0, 45], convention=0, degrees=True) # The person rotates of 45 degrees around the z axis of the train frame.

        If an application using an other convention required the rotation vector between the global frame and the person frame in the convention 4, you can access this rotation vector with the :meth:`get_global_rotation_vector` method:

        .. code-block:: python

            global_rotation_vector_person = person.get_global_rotation_vector(convention=4, degrees=True) # Get the rotation vector between the global frame and the person frame in convention 4 in degrees.

        """
        global_frame = self.get_global_frame()
        return global_frame.get_rotation_vector(convention=convention, degrees=degrees)

    def set_global_rotation_vector(self, rotation_vector: numpy.ndarray, *, convention: Optional[int] = None, degrees: bool = False) -> None:
        r"""
        Set the rotation vector representation of the rotation between the global frame and this frame in the given convention.

        The rotation vector must be a array-like with 3 elements representing the axis of rotation and the angle of rotation. The direction of the vector indicates the axis of rotation and the magnitude represents the angle of rotation.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - attribute :attr:`global_rotation_vector` to get or set the rotation vector between the global frame and this frame in the convention of the frame.
            - method :meth:`get_global_rotation_vector` to get the rotation vector between the global frame and this frame in a specific convention.
            - method :meth:`set_rotation_vector` to set the rotation vector between the parent frame and this frame in a specific convention.

        Parameters
        ----------
        rotation_vector : numpy.ndarray
            The rotation vector between the global frame and this frame in the given convention as an array-like with 3 elements.

        convention : Optional[int], optional
            Integer in ``[0, 7]`` selecting the convention. Defaults to the frame’s own convention.

        degrees : bool, optional
            If True, the rotation vector is in degrees. Default is False (radians).

        Examples
        --------
        Lets consider a train and a person standing into the train.

        First we can define the train frame relative to the global frame (the canonical frame) and then we can define the person frame relative to the train frame.

        .. code-block:: python

            from py3dframe import Frame

            train = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=None)
            person = Frame.from_axes(origin=[0, 1, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1], parent=train)

        Lets assume the train moves.

        .. code-block:: python

            train.set_translation([10, 0, 0], convention=0) # The train moves of 10 units along the x axis of the global frame.

        If an application using an other convention gives the new rotation vector between the global frame and the person frame in the convention 4, you can set this rotation vector with the :meth:`set_global_rotation_vector` method:

        .. code-block:: python

            import numpy

            rotation_vector = ... # Get the new rotation vector between the global frame and the person frame in convention 4 from the application.

            person.set_global_rotation_vector(rotation_vector, convention=4, degrees=True) # Set the rotation vector between the global frame and the person frame in convention 4 in degrees.

        The person frame is updated accordingly to keep the correct transformation between the global frame and the person frame.

        Then we can determine the new orientation and position of the person relative to the train frame:

        .. code-block:: python

            print("Person origin in the train frame:", person.origin)
            print("Person axes in the train frame:", person.axes)
        
        """
        if not isinstance(degrees, bool):
            raise TypeError("The degrees parameter must be a boolean.")
        rotation_vector = self._format_rotation_vector_dev(rotation_vector)
        R = Rotation.from_rotvec(rotation_vector, degrees=degrees)
        self.set_global_rotation(R, convention=convention)
    
    @property
    def global_rotation_vector(self) -> numpy.ndarray:
        r"""
        The rotation vector representation of the rotation between the global frame and this frame in the convention of the frame.

        The rotation vector describes the rotation using an axis of rotation and an angle of rotation in radians.

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`convention` to get or set the convention of the frame.
            - method :meth:`get_global_rotation_vector` to get the rotation vector in a specific convention.
            - method :meth:`set_global_rotation_vector` to set the rotation vector in a specific convention.

        Parameters
        ----------
        rotation_vector : numpy.ndarray
            The rotation vector between the global frame and this frame in the convention of the frame as an array-like with 3 elements. The rotation vector is in radians.

        Returns
        -------
        numpy.ndarray
            The rotation vector between the global frame and this frame in the convention of the frame with shape (3,). The rotation vector is in radians.

        """
        return self.get_global_rotation_vector(convention=self._convention, degrees=False)

    @global_rotation_vector.setter
    def global_rotation_vector(self, rotation_vector: numpy.ndarray) -> None:
        self.set_global_rotation_vector(rotation_vector, convention=self._convention, degrees=False)
    


    @property
    def global_axes(self) -> numpy.ndarray:
        r"""
        The basis vectors of the frame relative to the global frame.

        The axes is a 3x3 matrix with shape (3, 3) representing the basis vectors of the frame in the global frame coordinates.
        The first column is the x-axis, the second column is the y-axis and the third column is the z-axis.

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`global_x_axis` to access the x-axis of the frame.
            - attribute :attr:`global_y_axis` to access the y-axis of the frame.
            - attribute :attr:`global_z_axis` to access the z-axis of the frame.
            - attribute :attr:`axes` to access the basis vectors relative to the parent frame.

        Parameters
        ----------
        axes : numpy.ndarray
            The basis vectors of the frame in the global frame coordinates as an array-like with shape (3, 3).

        Returns
        -------
        numpy.ndarray
            The basis vectors of the frame in the global frame coordinates with shape (3, 3).
        """
        axes = self.get_global_rotation(convention=0).as_matrix()
        return axes

    @global_axes.setter
    def global_axes(self, axes: numpy.ndarray) -> None:
        axes = self._format_rotation_matrix_dev(axes)
        self.set_global_rotation_matrix(axes, convention=0)
    


    @property
    def global_x_axis(self) -> numpy.ndarray:
        r"""
        The x-axis of the frame relative to the global frame.

        The x-axis is a 3 elements vector with shape (3, 1) representing the coordinates of the x-axis of the frame in the global frame coordinates.

        .. note::

            This property is not settable. To change the x-axis, use the :attr:`global_axes` attribute.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`global_y_axis` to access the y-axis of the frame.
            - attribute :attr:`global_z_axis` to access the z-axis of the frame.
            - attribute :attr:`x_axis` to access the x-axis of the frame in the parent frame coordinates.

        Returns
        -------
        numpy.ndarray
            The x-axis of the frame in the global frame coordinates with shape (3, 1).
        """
        x_axis = self.get_global_rotation(convention=0).as_matrix()[:,0].reshape((3,1))
        return x_axis
    

    
    @property
    def global_y_axis(self) -> numpy.ndarray:
        r"""
        The y-axis of the frame relative to the global frame.

        The y-axis is a 3 elements vector with shape (3, 1) representing the coordinates of the y-axis of the frame in the global frame coordinates.

        .. note::

            This property is not settable. To change the y-axis, use the :attr:`global_axes` attribute.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`global_x_axis` to access the x-axis of the frame.
            - attribute :attr:`global_z_axis` to access the z-axis of the frame.
            - attribute :attr:`y_axis` to access the y-axis of the frame in the parent frame coordinates.

        Returns
        -------
        numpy.ndarray
            The y-axis of the frame in the global frame coordinates with shape (3, 1).
        """
        y_axis = self.get_global_rotation(convention=0).as_matrix()[:,1].reshape((3,1))
        return y_axis



    @property
    def global_z_axis(self) -> numpy.ndarray:
        r"""
        The z-axis of the frame relative to the global frame.

        The z-axis is a 3 elements vector with shape (3, 1) representing the coordinates of the z-axis of the frame in the global frame coordinates.

        .. note::

            This property is not settable. To change the z-axis, use the :attr:`global_axes` attribute.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :attr:`global_x_axis` to access the x-axis of the frame.
            - attribute :attr:`global_y_axis` to access the y-axis of the frame.
            - attribute :attr:`z_axis` to access the z-axis of the frame in the parent frame coordinates.

        Returns
        -------
        numpy.ndarray
            The z-axis of the frame in the global frame coordinates with shape (3, 1).
        """
        z_axis = self.get_global_rotation(convention=0).as_matrix()[:,2].reshape((3,1))
        return z_axis


    @property
    def global_origin(self) -> numpy.ndarray:
        r"""
        The origin of the frame relative to the global frame.

        The origin is a 3 elements vector with shape (3, 1) representing the coordinates of the origin of the frame in the global frame coordinates.

        .. note::

            This property is settable.

        .. seealso::

            - attribute :attr:`parent` to get or set the parent frame.
            - attribute :meth:`origin` to access the origin of the frame in the parent frame coordinates.

        Parameters
        ----------
        origin : numpy.ndarray
            The origin of the frame in the global frame coordinates as an array-like with 3 elements.

        Returns
        -------
        numpy.ndarray
            The origin of the frame in the global frame coordinates with shape (3, 1).
            
        """
        origin = self.get_global_translation(convention=0)
        return origin
    
    @global_origin.setter
    def global_origin(self, origin: numpy.ndarray) -> None:
        origin = self._format_translation_dev(origin)
        self.set_global_translation(origin, convention=0)

    # ====================================================================================================================================
    # Magic methods
    # ====================================================================================================================================
    def __repr__(self) -> str:
        r"""
        Return the string representation of the Frame object.

        Returns
        -------
        str
            The string representation of the Frame object.
        """
        global_frame = self.get_global_frame()
        global_origin = global_frame.origin.flatten()
        global_x_axis = global_frame.x_axis.flatten()
        global_y_axis = global_frame.y_axis.flatten()
        global_z_axis = global_frame.z_axis.flatten()
        return f"Frame(origin=[[{global_origin[0]}] [{global_origin[1]}] [{global_origin[2]}]], x_axis=[[{global_x_axis[0]}] [{global_x_axis[1]}] [{global_x_axis[2]}]], y_axis=[[{global_y_axis[0]}] [{global_y_axis[1]}] [{global_y_axis[2]}]], z_axis=[[{global_z_axis[0]}] [{global_z_axis[1]}] [{global_z_axis[2]}]])"


    def __eq__(self, other: Frame) -> bool:
        r"""
        Return the equality of the Frame object.
        
        Two Frame objects are equal if their global coordinates are equal.

        Parameters
        ----------
        other : Frame
            The other Frame object to compare.
        
        Returns
        -------
        bool
            True if the Frame objects are equal, False otherwise.
        """
        if not isinstance(other, Frame):
            return False
        global_frame = self.get_global_frame()
        other_global_frame = other.get_global_frame()
        return numpy.allclose(global_frame._T_dev, other_global_frame._T_dev) and numpy.allclose(global_frame._R_dev.as_quat(), other_global_frame._R_dev.as_quat())
    

    def __ne__(self, other: Frame) -> bool:
        r"""
        Return the inequality of the Frame object.
        
        Two Frame objects are equal if their global coordinates are equal.

        Parameters
        ----------
        other : Frame
            The other Frame object to compare.
        
        Returns
        -------
        bool
            True if the Frame objects are not equal, False otherwise.
        """
        return not self.__eq__(other)
    
    def copy(self) -> Frame:
        r"""
        Return a copy of the Frame object. Same parent frame (shallow copy).

        .. note::

            Only the frame itself is copied. The parent frame is not copied and is shared between the original and the copy.

        .. seealso::

            - method :meth:`deepcopy` to create a deep copy of the Frame object (including the parent frame).

        Returns
        -------
        Frame
            A copy of the Frame object.
        """
        new_frame = Frame(translation=self._T_dev.copy(), rotation=copy.deepcopy(self._R_dev), convention=0, parent=None)
        new_frame._parent = self._parent  # Shallow copy of the parent
        new_frame.convention = self._convention # Use the setter to ensure convention is valid
        return new_frame

    def deepcopy(self) -> Frame:
        r"""
        Return a deep copy of the Frame object. Deep copy of the parent frame.

        .. note::

            The parent frame is also copied recursively.

        .. seealso::

            - method :meth:`copy` to create a shallow copy of the Frame object (the parent frame is shared).

        Returns
        -------
        Frame
            A deep copy of the Frame object.
        """
        new_frame = Frame(translation=self._T_dev.copy(), rotation=copy.deepcopy(self._R_dev), convention=0, parent=None)
        new_frame._parent = None if self._parent is None else self._parent.deepcopy()  # Deep copy of the parent
        new_frame.convention = self._convention # Use the setter to ensure convention is valid
        return new_frame


    # ====================================================================================================================================
    # Load and Save method
    # ====================================================================================================================================
    def save_to_dict(self, method: Union[str, Sequence[str]] = ["quaternion", "rotation_vector", "rotation_matrix"]) -> Dict[str, Any]:
        r"""
        Save the Frame object to a dictionary.

        The dictionary has the following structure:

        .. code-block:: python

            {
                "translation": [float, float, float],
                "quaternion": [float, float, float, float],
                "rotation_vector": [float, float, float],
                "rotation_matrix": [[float, float, float], [float, float, float], [float, float, float]],
                "euler_angles": [float, float, float],
                "convention": int
                "parent": None
            }

        - The quaternion is given in WXYZ format (scalar first).
        - The rotation vector is given in radians.
        - The Euler angles are given in radians and the axes are "xyz".
        - The rotation is given in the convention of the frame.
        - The translation vector is given in the convention of the frame.

        .. seealso::

            - method :meth:`load_from_dict` to load the Frame object from a dictionary.

            For the reader, only one of the rotation keys is needed to reconstruct the frame. The other keys are provided for convenience and user experience.
            The reader chooses the key to use in the following order of preference if several are given:

            - quaternion
            - rotation_vector
            - rotation_matrix
            - euler_angles

        .. warning::

            ``euler_angles`` can raise a this warning : 
            
            - UserWarning: Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.

            I recommand to not use it.

        .. note::

            For retrocompatibility, the default method "quaternion" must be used.

        Parameters
        ----------
        method : Union[str, Sequence[str]], optional
            The method to use to save the rotation. It can be one of the following : "quaternion", "rotation_vector", "rotation_matrix" or "euler_angles".
            Several methods can be used at the same time. Default is ["quaternion", "rotation_vector", "rotation_matrix"].

        Returns
        -------
        Dict[str, Any]
            The dictionary containing the Frame object.
        """
        # Check if the method is a string or a list of strings
        if isinstance(method, str):
            method = [method]
        if not isinstance(method, Sequence):
            raise TypeError("The method must be a string or a list of strings.")
        if not all(isinstance(m, str) and m in ["quaternion", "rotation_vector", "rotation_matrix", "euler_angles"] for m in method):
            raise ValueError("The method must be one of the following : 'quaternion', 'rotation_vector', 'rotation_matrix' or 'euler_angles'.")

        data = {
            "translation": self.translation.flatten().tolist(),
            "convention": self._convention,
        }

        # Add the rotation method to the dictionary
        for m in method:
            if m == "quaternion":
                data["quaternion"] = self.get_quaternion(convention=None, scalar_first=True).tolist()
            elif m == "rotation_vector":
                data["rotation_vector"] = self.get_rotation_vector(convention=None, degrees=False).tolist()
            elif m == "rotation_matrix":
                data["rotation_matrix"] = self.get_rotation_matrix(convention=None).tolist()
            elif m == "euler_angles":
                data["euler_angles"] = self.get_euler_angles(convention=None, degrees=False, seq="xyz").tolist()

        # Add the parent frame to the dictionary
        if self._parent is None:
            data["parent"] = None
        else:
            data["parent"] = self._parent.save_to_dict(method=method)
        return data



    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> Frame:
        r"""
        Load the Frame object from a dictionary.

        The dictionary has the following structure:

        .. code-block:: python

            {
                "translation": [float, float, float],
                "quaternion": [float, float, float, float],
                "rotation_vector": [float, float, float],
                "rotation_matrix": [[float, float, float], [float, float, float], [float, float, float]],
                "euler_angles": [float, float, float],
                "convention": int
                "parent": None
            }

        - The quaternion is given in WXYZ format (scalar first).
        - The rotation vector is given in radians.
        - The Euler angles are given in radians and the axes are "xyz".
        - The rotation is given in the convention of the frame.
        - The translation vector is given in the convention of the frame.

        .. seealso::

            - method :meth:`save_to_dict` to save the Frame object to a dictionary.

        .. note::

            Only one of the rotation keys is needed to reconstruct the frame. 
            The reader chooses the key to use in the following order of preference if several are given:

            - quaternion
            - rotation_vector
            - rotation_matrix
            - euler_angles

        .. warning::

            ``euler_angles`` can raise a this warning : 
            
            - UserWarning: Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.

            I recommand to not use it.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary containing the Frame object.

        Returns
        -------
        Frame
            The Frame object.
        """
        # Check if the data is a dictionary and contains the required keys
        if not isinstance(data, dict):
            raise TypeError("The data must be a dictionary.")
        if not "translation" in data:
            raise ValueError("The dictionary must contain the 'translation' key.")
        if not "quaternion" in data and not "rotation_vector" in data and not "rotation_matrix" in data and not "euler_angles" in data:
            raise ValueError("The dictionary must contain at least one of the 'quaternion', 'rotation_vector', 'rotation_matrix' or 'euler_angles' keys.")
        if not "convention" in data:
            raise ValueError("The dictionary must contain the 'convention' key.")
        if not "parent" in data:
            raise ValueError("The dictionary must contain the 'parent' key.")
        # Convert the data to the correct types
        translation = numpy.asarray(data["translation"]).reshape((3,1)).astype(numpy.float64)
        convention = data["convention"]
        parent = data["parent"]
        if parent is None:
            parent_frame = None
        else:
            parent_frame = cls.load_from_dict(parent)
        # According to the order of preference, create the rotation object
        if "quaternion" in data:
            quaternion = numpy.asarray(data["quaternion"]).reshape((4,)).astype(numpy.float64)
            rotation = Rotation.from_quat(quaternion, scalar_first=True)
        elif "rotation_vector" in data:
            rotation_vector = numpy.asarray(data["rotation_vector"]).reshape((3,)).astype(numpy.float64)
            rotation = Rotation.from_rotvec(rotation_vector, degrees=False)
        elif "rotation_matrix" in data:
            rotation_matrix = numpy.asarray(data["rotation_matrix"]).astype(numpy.float64)
            rotation = Rotation.from_matrix(rotation_matrix)
        elif "euler_angles" in data:
            euler_angles = numpy.asarray(data["euler_angles"]).reshape((3,)).astype(numpy.float64)
            rotation = Rotation.from_euler("xyz", euler_angles, degrees=False)
        else:
            raise ValueError("The dictionary must contain at least one of the 'quaternion', 'rotation_vector', 'rotation_matrix' or 'euler_angles' keys.")
        # Create the Frame object
        frame = cls.from_rotation(translation=translation, rotation=rotation, parent=parent_frame, convention=convention)
        return frame



    def save_to_json(self, filename: str) -> None:
        """
        Save the Frame object to a JSON file.

        .. seealso::

            - method :meth:`save_to_dict` to save the Frame object to a dictionary.

        Parameters
        ----------
        filename : str
            The name of the JSON file.
        """
        data = self.save_to_dict()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    


    @classmethod
    def load_from_json(cls, filename: str) -> Frame:
        """
        Load the Frame object from a JSON file.

        .. seealso::

            - method :meth:`load_from_dict` to load the Frame object from a dictionary.

        Parameters
        ----------
        filename : str
            The name of the JSON file.
        
        Returns
        -------
        Frame
            The Frame object.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        frame = cls.load_from_dict(data)
        return frame