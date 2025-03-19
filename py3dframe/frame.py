from __future__ import annotations
from typing import Optional, Dict
import numpy
import uuid
import json
from scipy.spatial.transform import Rotation



class Frame(object):
    r"""
    Represents a orthonormal reference frame in :math:`\mathbb{R}^3`.

    The frame is defined by a origin and an orientation.
    The orientation can be defined using a rotation matrix, quaternion, Euler angles, or a rotation vector.
    The frame can be direct or indirect, depending on the determinant of the rotation matrix.
    The convention for quaternion is scalar first :math:`[w, x, y, z]`.

    The frame use scipy.spatial.transform.Rotation to manage the rotation matrix.

    Convention for the frame :

    Lets consider a first frame :math:`E = (\mathbf{O}, \mathbf{\vec{e_x}}, \mathbf{\vec{e_y}}, \mathbf{\vec{e_z}})`.
    Lets consider a second frame :math:`F = (\mathbf{C}, \mathbf{\vec{i}}, \mathbf{\vec{j}}, \mathbf{\vec{k}})`.

    The transformation from the frame E (parent frame) to the frame F (child frame) can be stored in a Frame object.
    To construct the Frame object, the user must provide the origin and the orientation of the child frame expressed in the parent frame coordinates.
    The `origin` is the coordinates of the origin of the child frame in the parent frame coordinates.
    The `rotation_matrix` is constructed as follow:

    .. math::

        \mathbf{R} = \begin{bmatrix} \mathbf{\vec{i}} & \mathbf{\vec{j}} & \mathbf{\vec{k}} \end{bmatrix}

    For example, to construct the frame where teh x and y axis are inverted, the user can use the following code:

    .. code-block:: python

        origin = numpy.array([0.0, 0.0, 0.0]) # Express the origin of the child frame in the parent frame coordinates
        x_axis = numpy.array([0.0, 1.0, 0.0]) # Express the x-axis of the child frame in the parent frame coordinates
        y_axis = numpy.array([1.0, 0.0, 0.0]) # Express the y-axis of the child frame in the parent frame coordinates
        z_axis = numpy.array([0.0, 0.0, 1.0]) # Express the z-axis of the child frame in the parent frame coordinates
        rotation_matrix = numpy.array([x_axis, y_axis, z_axis]) # Construct the rotation matrix
        frame = Frame(origin=origin, rotation_matrix=rotation_matrix, parent=None)

    With this convention, if we consider a point :math:`\mathbf{p}` and we note 
    :math:`\mathbf{p}_{\text{parent}}` the coordinates of the point in the parent frame (:math:`E`), 
    and :math:`\mathbf{p}_{\text{frame}}` the coordinates of the point in the child frame (:math:`F`),
    the equation between :math:`\mathbf{p}_{\text{parent}}` and :math:`\mathbf{p}_{\text{frame}}` is:

    .. math::

        \mathbf{p}_{\text{parent}} = \mathbf{R} \mathbf{p}_{\text{frame}} + \mathbf{O}

    where :math:`\mathbf{R}` and :math:`\mathbf{O}` are the rotation matrix and the origin of the Frame object between the parent frame and the child frame.

    We have the reverse equation to convert a point from the child frame to the parent frame:

    .. math::

        \mathbf{p}_{\text{frame}} = \mathbf{R}^T (\mathbf{p}_{\text{parent}} - \mathbf{O})

    When the rotation matrix is set, the determinant is checked to determine if the frame is direct or indirect.

    However when the user create the frame using the quaternion or the euler angles, the frame is always considered as direct.
    Then the user must set the frame as indirect using the set_indirect method.
    In this case the Z-axis is inverted to have a right-handed frame.

    In fact when the frame is indirect, the user can : 

    - set directly the indirect rotation matrix.
    - set the quaternion or the euler angles of the associated direct frame and then set the frame as indirect.

    .. seealso:: 
    
        :class:`py3dframe.FrameTree` To manage easily multiple frames and their relationships.

    Parameters
    ----------
    origin : numpy.ndarray, optional
        The origin of the frame in 3D space with shape (3,1) expressed in the parent frame coordinates.
        The default is None - [0.0, 0.0, 0.0].
    
    quaternion : numpy.ndarray, optional
        The quaternion of the frame with shape (4,) in scalar first :math:`[w, x, y, z]` convention.
        The default is None - [1.0, 0.0, 0.0, 0.0].
    
    rotation_matrix : numpy.ndarray, optional
        The rotation matrix of the frame with shape (3,3) expressed in the parent frame coordinates.
        The default is None - identity matrix.

    O3_project : bool, optional
        If True, the input rotation matrrix is projected
    
    euler_angles : numpy.ndarray, optional
        The Euler angles of the frame in radians with shape (3,). The default is None.
    
    rotation_vector : numpy.ndarray, optional 
        The rotation vector of the frame with shape (3,). The default is None.
    
    direct : bool, optional
        Set if the frame is direct or indirect. The default is True.

    parent : Frame, optional
        The parent frame of the current frame. The default is None.

    _uuid : uuid.UUID, optional
        The unique identifier of the frame. The default is None. This parameter is used by :class:`py3dframe.FrameTree`, do not use it.
        This parameter is used to identify the frame in the FrameTree and it is not saved in the JSON file.


    Examples
    ---------

    Example of using the `Frame` class to obtain its rotation matrix.

    .. code-block:: python

        frame = Frame(
            origin=numpy.array([1.0, 2.0, 3.0]), 
            quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), 
            direct=True
        )
        print(frame.rotation_matrix)

    """

    _tolerance = 1e-6

    # Initialization
    def __init__(
        self,
        *,
        origin: Optional[numpy.ndarray] = None,
        quaternion: Optional[numpy.ndarray] = None,
        rotation_matrix: Optional[numpy.ndarray] = None,
        O3_project: bool = False,
        euler_angles: Optional[numpy.ndarray] = None,
        rotation_vector: Optional[numpy.ndarray] = None,
        direct: bool = True,
        parent: Optional[Frame] = None,
        _uuid: Optional[uuid.UUID] = None
        ) -> None:
        # Check if only one of the orientation parameters is provided
        if sum([quaternion is not None, rotation_matrix is not None, euler_angles is not None, rotation_vector is not None]) > 1:
            raise ValueError("Only one of 'quaternion', 'rotation_matrix', 'euler_angles' and 'rotation_vector' can be provided.")
        
        # Default values if no orientation is provided
        elif sum([quaternion is not None, rotation_matrix is not None, euler_angles is not None, rotation_vector is not None]) == 0:
            quaternion = numpy.array([1.0, 0.0, 0.0, 0.0]).astype(float)
        if origin is None:
            origin = numpy.zeros((3,1)).astype(float)

        # Initialize the frame
        self.origin = origin
        self.direct = direct
        self.parent = parent
        self._uuid = _uuid
        if quaternion is not None:
            self.quaternion = quaternion
        elif rotation_matrix is not None:
            if O3_project:
                self.set_O3_rotation_matrix(rotation_matrix)
            else:
                self.rotation_matrix = rotation_matrix
        elif euler_angles is not None:
            self.euler_angles = euler_angles
        elif rotation_vector is not None:
            self.rotation_vector = rotation_vector


    # Properties getters and setters
    @property
    def origin(self) -> numpy.ndarray:
        r"""
        Get or set the origin of the frame in 3D space with shape (3,1).

        The origin is the coordinates of the frame in the global frame.

        Parameters
        ----------
        origin : numpy.ndarray
            The origin of the frame with shape (3,1).
        
        Raises
        ------
        TypeError
            If the origin is not a numpy array with shape (3,).
        """
        return self._origin
    
    @origin.setter
    def origin(self, origin: numpy.ndarray) -> None:
        self._origin = numpy.array(origin).reshape((3,1)).astype(float)
    


    @property
    def quaternion(self) -> numpy.ndarray:
        r"""
        Get or set the quaternion of the frame.

        The convention for quaternion is scalar first :math:`[w, x, y, z]`.
        The quaternion is normalized to have a unit magnitude.

        .. warning::

            The direct property is set to True when the quaternion is set.
            The user must set the direct property to False if the quaternion is for the associated indirect frame.

        .. seealso::

            :meth:`py3dframe.Frame.set_quaternion` method to set the quaternion, the direct property, and the convention.
            :meth:`py3dframe.Frame.get_quaternion` method to get the quaternion with a given convention.

        Parameters
        ----------
        quaternion : numpy.ndarray
            The quaternion of the frame with shape (4,) and scalar first :math:`[w, x, y, z]` convention.
        
        Raises
        ------
        ValueError
            If the quaternion has zero magnitude.
        TypeError
            If the quaternion is not a numpy array with shape (4,).
        """
        return self.get_quaternion(scalar_first=True)
    
    @quaternion.setter
    def quaternion(self, quaternion: numpy.ndarray) -> None:
        self.set_quaternion(quaternion, scalar_first=True, direct=True)

    def set_quaternion(self, quaternion: numpy.ndarray, scalar_first: bool = True, direct: bool = True) -> None:
        r"""
        Set the quaternion of the frame and the direct property.

        The quaternion is normalized to have a unit magnitude.

        Parameters
        ----------
        quaternion : numpy.ndarray
            The quaternion of the frame with shape (4,).
        
        scalar_first : bool
            Set the scalar first convention of the quaternion. The default is True.
        
        direct : bool
            Set the frame as direct or indirect. The default is True.
        
        Raises
        ------
        ValueError
            If the quaternion has zero magnitude.
        TypeError
            If the quaternion is not a numpy array with shape (4,).
        """
        quaternion = numpy.array(quaternion).reshape((4,)).astype(float)
        norm = numpy.linalg.norm(quaternion, ord=None)
        if abs(norm) < self._tolerance:
            raise ValueError("Quaternion cannot have zero magnitude.")
        quaternion = quaternion / norm
        self._rotation = Rotation.from_quat(quaternion, scalar_first=scalar_first)
        self.direct = direct

    def get_quaternion(self, scalar_first: bool = True) -> numpy.ndarray:
        """
        Get the quaternion of the frame with a given convention.

        Parameters
        ----------
        scalar_first : bool
            Set the scalar first convention of the quaternion. The default is True.
        
        Returns
        -------
        numpy.ndarray
            The quaternion of the frame with shape (4,).
        """
        quaternion = self._rotation.as_quat(scalar_first=scalar_first)
        quaternion = quaternion / numpy.linalg.norm(quaternion, ord=None)
        return quaternion
    


    @property
    def rotation_matrix(self) -> numpy.ndarray:
        r"""
        Get or set the rotation matrix of the frame.

        The rotation matrix is normalized to have a determinant of 1.
        The frame is considered as direct if the determinant is 1.
        The frame is considered as indirect if the determinant is -1.

        .. note::

            The direct property is changed when the rotation matrix is set.

        To be sure that the rotation matrix is in the orthogonal group :math:`O(3)`, the user can use the method :meth:`py3dframe.Frame.set_O3_rotation_matrix`.

        Parameters
        ----------
        rotation_matrix : numpy.ndarray
            The rotation matrix of the frame with shape (3,3).
        
        Raises
        ------
        ValueError
            If the rotation matrix has a determinant different from 1 or -1.
        TypeError
            If the rotation matrix is not a numpy array with shape (3,3).
        """
        rotation_matrix = self._rotation.as_matrix()
        if not self.direct:
            rotation_matrix[:, 2] = -rotation_matrix[:, 2]
        return rotation_matrix

    @rotation_matrix.setter
    def rotation_matrix(self, rotation_matrix: numpy.ndarray) -> None:
        rotation_matrix = numpy.array(rotation_matrix).reshape((3,3)).astype(float)
        det = numpy.linalg.det(rotation_matrix)
        # Check if the determinant is close to 1
        if abs(abs(det) - 1.0) > self._tolerance:
            raise ValueError("Rotation matrix must have a determinant of 1 or -1.")
        # Check if the frame is direct or indirect
        if det < 0:
            rotation_matrix[:, 2] = -rotation_matrix[:, 2] # Invert the Z-axis to have a right-handed frame
            self.direct = False
        else:
            self.direct = True
        self._rotation = Rotation.from_matrix(rotation_matrix)



    @property
    def euler_angles(self) -> numpy.ndarray:
        r"""
        Get or set the Euler angles of the frame.

        The Euler angles are in radians and in XYZ convention.

        .. warning::

            The direct property is set to True when the Euler angles are set.
            The user must set the direct property to False if the Euler angles are for the associated indirect frame.

        .. seealso::

            :meth:`py3dframe.Frame.set_euler_angles` method to set the Euler angles, the direct property, and the convention.
            :meth:`py3dframe.Frame.get_euler_angles` method to get the Euler angles with a given convention.

        Parameters
        ----------
        euler_angles : numpy.ndarray
            The Euler angles of the frame in radians with shape (3,) and XYZ convention.
        
        Raises
        ------
        TypeError
            If the Euler angles are not a numpy array with shape (3,).
        """
        return self.get_euler_angles(degrees=False, axes="XYZ", direct=True)

    @euler_angles.setter
    def euler_angles(self, euler_angles: numpy.ndarray) -> None:
        self.set_euler_angles(euler_angles, degrees=False, axes="XYZ", direct=True)
    
    def set_euler_angles(self, euler_angles: numpy.ndarray, degrees: bool = False, axes: str = "XYZ", direct: bool = True) -> None:
        r"""
        Set the Euler angles of the frame and the direct property.

        The axes convention can be set to XYZ, ZYX, ZXY, YXZ, YZX, or XZY.

        Parameters
        ----------
        euler_angles : numpy.ndarray
            The Euler angles of the frame in radians with shape (3,). 

        degrees : bool
            Set if the Euler angles are in degrees. The default is False.
        
        axes : str
            Set the axes convention of the Euler angles. The default is "XYZ".
        
        direct : bool
            Set the frame as direct or indirect. The default is True.
        
        Raises
        ------
        TypeError
            If the Euler angles are not a numpy array with shape (3,).
        """
        euler_angles = numpy.array(euler_angles).reshape((3,)).astype(float)
        self._rotation = Rotation.from_euler(axes, euler_angles, degrees=degrees)
        self.direct = direct
    
    def get_euler_angles(self, degrees: bool = False, axes: str = "XYZ") -> numpy.ndarray:
        r"""
        Get the Euler angles of the frame with a given convention.

        The axes convention can be set to XYZ, ZYX, ZXY, YXZ, YZX, or XZY.

        Parameters
        ----------
        degrees : bool
            Set if the Euler angles are in degrees. The default is False.
        
        axes : str
            Set the axes convention of the Euler angles. The default is "XYZ".
        
        Returns
        -------
        numpy.ndarray
            The Euler angles of the frame in radians with shape (3,).
        """
        euler_angles = self._rotation.as_euler(axes, degrees=degrees)
        return euler_angles
    


    @property
    def rotation_vector(self) -> numpy.ndarray:
        r"""
        Get or set the rotation vector of the frame.

        The rotation associted is defined by the Rodrigues formula:

        .. math::

            \mathbf{R} = \mathbf{I} + \sin(\theta) \mathbf{K} + (1 - \cos(\theta)) \mathbf{K}^2
        
        The associated rotation vector is:

        .. math::

            \mathbf{r} = \theta \mathbf{k}
        
        Where :math:`k` is the unit vector axis and :math:`\theta` is the rotation angle in radians.

        .. warning::

            The direct property is set to True when the rotation vector is set.
            The user must set the direct property to False if the rotation vector is for the associated indirect frame.

        .. seealso::

            :meth:`py3dframe.Frame.set_rotation_vector` method to set the rotation vector, the direct property, and the convention.
            :meth:`py3dframe.Frame.get_rotation_vector` method to get the rotation vector with a given convention.

        Parameters
        ----------
        rotation_vector : numpy.ndarray
            The rotation vector of the frame with shape (3,).
        
        Raises
        ------
        TypeError
            If the rotation vector is not a numpy array with shape (3,).
        """
        return self.get_rotation_vector(degrees=False, direct=True)

    @rotation_vector.setter
    def rotation_vector(self, rotation_vector: numpy.ndarray) -> None:
        self.set_rotation_vector(rotation_vector, degrees=False, direct=True)
    
    def set_rotation_vector(self, rotation_vector: numpy.ndarray, degrees: bool = False, direct: bool = True) -> None:
        r"""
        Set the rotation vector of the frame and the direct property.

        Parameters
        ----------
        rotation_vector : numpy.ndarray
            The rotation vector of the frame with shape (3,).
        
        degrees : bool
            Set if the rotation vector is in degrees. The default is False.
        
        direct : bool
            Set the frame as direct or indirect. The default is True.
        
        Raises
        ------
        TypeError
            If the rotation vector is not a numpy array with shape (3,).
        """
        rotation_vector = numpy.array(rotation_vector).reshape((3,)).astype(float)
        theta = numpy.linalg.norm(rotation_vector, ord=None)
        if abs(theta) < self._tolerance:
            rotation_vector = numpy.zeros((3,)).astype(float)
        else:
            rotation_vector = rotation_vector / theta
        self._rotation = Rotation.from_rotvec(rotation_vector, degrees=degrees)
        self.direct = direct
    
    def get_rotation_vector(self, degrees: bool = False) -> numpy.ndarray:
        r"""
        Get the rotation vector of the frame with a given convention.

        Parameters
        ----------
        degrees : bool
            Set if the rotation vector is in degrees. The default is False.
        
        Returns
        -------
        numpy.ndarray
            The rotation vector of the frame with shape (3,).
        """
        rotation_vector = self._rotation.as_rotvec(degrees=degrees)
        return rotation_vector



    @property
    def homogeneous_matrix(self) -> numpy.ndarray:
        r"""
        Get or set the homogeneous transformation matrix of the frame.

        The homogeneous matrix is a 4x4 matrix defined by:

        .. math::

            \begin{bmatrix}
                \mathbf{R} & \mathbf{p} \\
                0 & 1
            \end{bmatrix}
        
        Where :math:`\mathbf{R}` is the rotation matrix and :math:`\mathbf{p}` is the origin.

        .. note::

            The direct property is changed when the rotation matrix is set.

        Parameters
        ----------
        matrix : numpy.ndarray
            The homogeneous transformation matrix of the frame with shape (4,4).
        
        Raises
        ------
        TypeError
            If the matrix is not a numpy array.
        ValueError
            If the matrix is not 4x4.
        """
        matrix = numpy.eye(4)
        matrix[:3, :3] = self.rotation_matrix
        matrix[:3, 3] = self.origin
        return matrix

    @homogeneous_matrix.setter
    def homogeneous_matrix(self, matrix: numpy.ndarray) -> None:
        homogeneous_matrix = numpy.array(matrix).reshape((4,4)).astype(float)
        self.origin = homogeneous_matrix[:3, 3]
        self.rotation_matrix = homogeneous_matrix[:3, :3]



    @property
    def direct(self) -> bool:
        r"""
        Get or set if the frame is direct or indirect.

        The frame is direct if the determinant of the rotation matrix is 1.
        The frame is indirect if the determinant of the rotation matrix is -1.
    
        If indirect is set to True, the Z-axis of the frame is inverted to have a right-handed frame.

        Parameters
        ----------
        direct : bool
            Set the frame as direct or indirect.
        
        Raises
        ------
        ValueError
            If the direct is not a boolean.
        """
        if not isinstance(self._direct, bool):
            raise ValueError("Direct must be a boolean.")
        return self._direct
    
    @direct.setter
    def direct(self, direct: bool) -> None:
        if not isinstance(direct, bool):
            raise ValueError("Direct must be a boolean.")
        self._direct = direct



    @property
    def parent(self) -> Frame:
        r"""
        Get or set the parent frame of the current frame.

        The parent frame is the frame in which the current frame is defined.

        Parameters
        ----------
        parent : Frame
            The parent frame of the current frame.
        
        Raises
        ------
        TypeError
            If the parent is not a Frame.
        """
        return self._parent

    @parent.setter
    def parent(self, parent: Frame) -> None:
        if parent is not None and not isinstance(parent, Frame):
            raise TypeError("The parent must be a Frame.")
        self._parent = parent



    @property
    def x_axis(self) -> numpy.ndarray:
        """Get the X-axis of the frame with shape (3,1)"""
        x_axis = self.rotation_matrix[:, 0].reshape((3,1))
        x_axis = x_axis / numpy.linalg.norm(x_axis, ord=None)
        return x_axis
    


    @property
    def y_axis(self) -> numpy.ndarray:
        """Get the Y-axis of the frame with shape (3,1)"""
        y_axis = self.rotation_matrix[:, 1].reshape((3,1))
        y_axis = y_axis / numpy.linalg.norm(y_axis, ord=None)
        return y_axis



    @property
    def z_axis(self) -> numpy.ndarray:
        """Get the Z-axis of the frame with shape (3,1)"""
        z_axis = self.rotation_matrix[:, 2].reshape((3,1))
        z_axis = z_axis / numpy.linalg.norm(z_axis, ord=None)
        return z_axis



    @property
    def is_direct(self) -> bool:
        """Check if the frame is direct."""
        return self.direct
    


    @property
    def is_indirect(self) -> bool:
        """Check if the frame is indirect."""
        return not self.direct



    # Private methods
    def _O3_projection(self, matrix: numpy.ndarray) -> numpy.ndarray:
        r"""
        Project a matrix to the orthogonal group :math:`O(3)` using SVD and minimisation of the frobenius norm.

        The orthogonal group `O(3)` is the set of 3x3 matrices with determinant 1.

        To project a matrix to `O(3)`, the SVD is computed and the orthogonal matrix is obtained by:

        .. math::

            \mathbf{O} = \mathbf{U} \mathbf{V}^T
        
        where :math:`\mathbf{U}` and :math:`\mathbf{V}` are the left and right singular vectors of the matrix such as:

        .. math::

            \mathbf{M} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T

        Parameters
        ----------
        matrix : numpy.ndarray
            A 3x3 matrix to be projected.

        Returns
        -------
        numpy.ndarray
            The `O(3)` projection of the matrix.

        Raises
        ------
        TypeError
            If the matrix is not a numpy array.
        ValueError
            If the matrix is not 3x3.
        """
        # Check parameters
        if not isinstance(matrix, numpy.ndarray):
            raise TypeError("The matrix must be a numpy array.")
        if matrix.shape != (3, 3):
            raise ValueError("The matrix must be 3x3.")
    
        # Compute the SVD
        U, _, Vt = numpy.linalg.svd(matrix)
        orthogonal_matrix = numpy.dot(U, Vt)
        return orthogonal_matrix
    


    # Public methods
    def set_direct(self, direct: bool = True) -> None:
        r"""
        Set the frame as direct or indirect.

        .. seealso::

            :meth:`py3dframe.Frame.set_indirect` method to set the frame as indirect.
            :attr:`py3dframe.Frame.direct` property to get or set the direct property.

        Parameters
        ----------
        direct : bool, optional
            Set the frame as direct or indirect, by default True.
        
        Raises
        ------
        TypeError
            If the direct is not a boolean
        """
        # Check the parameter
        if not isinstance(direct, bool):
            raise TypeError("Direct must be a boolean.")
        
        # Set the direct property
        self.direct = direct
    


    def set_indirect(self, indirect: bool = True) -> None:
        r"""
        Set the frame as indirect or direct.

        .. seealso::

            :meth:`py3dframe.Frame.set_direct` method to set the frame as direct.
            :attr:`py3dframe.Frame.direct` property to get or set the direct property.

        Parameters
        ----------
        indirect : bool, optional
            Set the frame as indirect or direct, by default True.
        
        Raises
        ------
        TypeError
            If the indirect is not a boolean
        """
        # Check the parameter
        if not isinstance(indirect, bool):
            raise TypeError("Indirect must be a boolean.")

        # Set the direct property
        self.direct = not indirect



    def set_O3_rotation_matrix(self, matrix: numpy.ndarray) -> None:
        r"""
        Set the rotation matrix of the frame in :math:`O(3)`.

        The given matrix is projected to the orthogonal group :math:`O(3)` using the SVD method before setting the rotation matrix.

        .. seealso::

            :meth:`py3dframe.Frame._O3_projection` method to project a matrix to :math:`O(3)`.

        Parameters
        ----------
        matrix : numpy.ndarray
           The rotation matrix with shape (3,3) to be projected to :math:`O(3)`.
        
        Raises
        ------
        TypeError 
            If the matrix is not a numpy array.
        ValueError
            If the matrix is not 3x3.
        """
        # Check the matrix
        if not isinstance(matrix, numpy.ndarray):
            raise TypeError("The matrix must be a numpy array.")
        if matrix.shape != (3, 3):
            raise ValueError("The matrix must be 3x3.")
        
        # Project the matrix to O(3)
        rotation_matrix = self._O3_projection(matrix)
        self.rotation_matrix = rotation_matrix



    def get_global_frame(self) -> Frame:
        """
        Get the transformation between the global frame and the frame. 

        The transformation is defined by the Frame object between the global frame and the frame.

        If the frame has a parent, the global frame is defined by the composition of the parent frame and the current frame (and recursively).

        .. seealso::

            :meth:`py3dframe.Frame.compose` method to compose two frames.

        .. warning::

            The global frame is defined in the current geometry ! If the current frame or a parent is modified, the global frame will not be affected.

        Returns
        -------
        Frame
            The transformation between the global frame and the frame.
        """
        global_frame = self
        while global_frame.parent is not None:
            parent = global_frame.parent
            global_frame = parent * global_frame
        return global_frame



    def from_parent_to_frame(
        self,
        *,
        point: Optional[numpy.ndarray] = None,
        vector: Optional[numpy.ndarray] = None,
        transpose: bool = False
        ) -> Optional[numpy.ndarray]:
        r"""
        Convert a point or vector from parent frame coordinates to frame coordinates.

        For a point, the equation is:

        .. math::

            \mathbf{p}_{\text{frame}} = \mathbf{R}^T (\mathbf{p}_{\text{parent}} - \mathbf{O})

        For a vector, the equation is:

        .. math::

            \mathbf{v}_{\text{frame}} = \mathbf{R}^T \mathbf{v}_{\text{parent}}

        if point is not None and vector is not None, the function will raise a ValueError.
        if point is None and vector is None, the function will return None.

        .. seealso::

            - :meth:`py3dframe.Frame.from_frame_to_parent` method to convert a point or vector from frame coordinates to parent frame coordinates.
            - :meth:`py3dframe.Frame.from_global_to_frame` method to convert a point or vector from global coordinates to frame coordinates.

        Parameters
        ----------
        point : numpy.ndarray, optional
            Point in parent frame coordinates with shape (3, N), by default None.
        
        vector : numpy.ndarray, optional
            Vector in parent frame coordinates with shape (3, N), by default None.
        
        transpose : bool, optional
            Set if the input points or vectors are given in (N, 3) shape. The default is False.
        
        Returns
        -------
        numpy.ndarray
            Point or vector in frame coordinates with shape (3, N), (or (N, 3) if transpose is True).

        Raises
        ------
        ValueError
            If both point and vector are provided.

        Examples
        --------

        >>> parent = Frame(origin=numpy.array([1.0, -1.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True, parent=parent)
        >>> point_parent = numpy.array([[1.0], [2.0], [3.0]])
        >>> point_frame = frame.from_parent_to_frame(point=point_parent)

        """
        if point is not None and vector is not None:
            raise ValueError("Only one of 'point' or 'vector' can be provided.")
        if point is None and vector is None:
            return None
        
        parent_data = point if point is not None else vector

        # Convert the data into (3, N) shape
        if transpose:
            parent_data = numpy.array(parent_data).reshape((-1, 3)).astype(float).T
        else:
            parent_data = numpy.array(parent_data).reshape((3, -1)).astype(float)
        
        # Convert the point to vector
        if point is not None:
            parent_data = parent_data - self.origin
        
        # Convert the data to frame coordinates
        frame_data = numpy.dot(self.rotation_matrix.T, parent_data)

        # Organize the output array
        if transpose:
            frame_data = frame_data.T
        return frame_data
    


    def from_frame_to_parent(
        self,
        *,
        point: Optional[numpy.ndarray] = None,
        vector: Optional[numpy.ndarray] = None,
        transpose: bool = False
        ) -> Optional[numpy.ndarray]:
        r"""
        Convert a point or vector from frame coordinates to parent frame coordinates.

        For a point, the equation is:

        .. math::

            \mathbf{p}_{\text{parent}} = \mathbf{R} \mathbf{p}_{\text{frame}} + \mathbf{O}
        
        For a vector, the equation is:

        .. math::

            \mathbf{v}_{\text{parent}} = \mathbf{R} \mathbf{v}_{\text{frame}}  

        if point is not None and vector is not None, the function will raise a ValueError.
        if point is None and vector is None, the function will return None.

        .. seealso::

            - :meth:`py3dframe.Frame.from_parent_to_frame` method to convert a point or vector from parent frame coordinates to frame coordinates.
            - :meth:`py3dframe.Frame.from_frame_to_global` method to convert a point or vector from frame coordinates to global coordinates.

        Parameters
        ----------
        point : numpy.ndarray, optional
            Point in frame coordinates with shape (3, N), by default None.
        
        vector : numpy.ndarray, optional
            Vector in frame coordinates with shape (3, N), by default None.
        
        transpose : bool, optional
            Set if the input points or vectors are given in (N, 3) shape. The default is False.
        
        Returns
        -------
        numpy.ndarray
            Point or vector in parent frame coordinates with shape (3, N), (or (N, 3) if transpose is True).    
        
        Raises
        ------
        ValueError
            If both point and vector are provided.
        
        Examples
        --------

        >>> parent = Frame(origin=numpy.array([1.0, -1.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True, parent=parent
        >>> point_frame = numpy.array([[1.0], [2.0], [3.0]])
        >>> point_parent = frame.from_frame_to_parent(point=point_frame)     

        """
        if point is not None and vector is not None:
            raise ValueError("Only one of 'point' or 'vector' can be provided.")
        if point is None and vector is None:
            return None
        
        frame_data = point if point is not None else vector

        # Convert the data into (3, N) shape
        if transpose:
            frame_data = numpy.array(frame_data).reshape((-1, 3)).astype(float).T
        else:
            frame_data = numpy.array(frame_data).reshape((3, -1)).astype(float)
        
        # Convert the data to parent frame coordinates
        parent_data = numpy.dot(self.rotation_matrix, frame_data)

        # Convert the vector to point
        if point is not None:
            parent_data = parent_data + self.origin
        
        # Organize the output array
        if transpose:
            parent_data = parent_data.T
        return parent_data



    def from_global_to_frame(
        self, 
        *, 
        point: Optional[numpy.ndarray] = None, 
        vector: Optional[numpy.ndarray] = None,
        transpose: bool = False
        ) -> Optional[numpy.ndarray]:
        r"""
        Convert a point or vector from global coordinates to frame coordinates.

        This method get the global frame of the frame and convert the point or vector to the frame coordinates.

        Lets note :math:`\mathbf{R_{(G)}}` and :math:`\mathbf{O_{(G)}}` the rotation matrix and the origin of the Frame object between the global frame and the frame.

        For a point, the equation is:

        .. math::

            \mathbf{p}_{\text{frame}} = \mathbf{R_{(G)}}^T (\mathbf{p}_{\text{global}} - \mathbf{O_{(G)})

        For a vector, the equation is:

        .. math:: 

            \mathbf{v}_{\text{frame}} = \mathbf{R_{(G)}}^T \mathbf{v}_{\text{global}}

        if point is not None and vector is not None, the function will raise a ValueError.
        if point is None and vector is None, the function will return None.

        .. seealso::

            - :meth:`py3dframe.Frame.from_frame_to_global` method to convert a point or vector from frame coordinates to global coordinates.
            - :meth:`py3dframe.Frame.from_parent_to_frame` method to convert a point or vector from parent frame coordinates to frame coordinates.

        Parameters
        ----------
        point : numpy.ndarray, optional
            Point in global coordinates with shape (3, N), by default None.
        
        vector : numpy.ndarray, optional
            Vector in global coordinates with shape (3, N), by default None.

        transpose : bool, optional
            Set if the input points or vectors are given in (N, 3) shape. The default is False.
        
        Returns
        -------
        numpy.ndarray
            Point or vector in frame coordinates with shape (3, N) (or (N, 3) if transpose is True).
        
        Examples
        --------

        >>> parent = Frame(origin=numpy.array([1.0, -1.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True, parent=parent)
        >>> point_global = numpy.array([[1.0], [2.0], [3.0]])
        >>> point_frame = frame.from_global_to_frame(point=point_global)

        """
        global_frame = self.get_global_frame()
        return global_frame.from_parent_to_frame(point=point, vector=vector, transpose=transpose)



    def from_frame_to_global(
        self, 
        *, 
        point: Optional[numpy.ndarray] = None,
        vector: Optional[numpy.ndarray] = None,
        transpose: bool = False
        ) -> Optional[numpy.ndarray]:
        r"""
        Convert a point or vector from frame coordinates to global coordinates.

        This method get the global frame of the frame and convert the point or vector to the global coordinates.

        Lets note :math:`\mathbf{R_{(G)}}` and :math:`\mathbf{O_{(G)}}` the rotation matrix and the origin of the Frame object between the global frame and the frame.

        For a point, the equation is:

        .. math::

            \mathbf{p}_{\text{global}} = \mathbf{R_{(G)}} (\mathbf{p}_{\text{frame}}) + \mathbf{O_{(G)}}
        
        For a vector, the equation is:

        .. math::

            \mathbf{v}_{\text{global}} = \mathbf{R_{(G)}} \mathbf{v}_{\text{frame}}

        if point is not None and vector is not None, the function will raise a ValueError.
        if point is None and vector is None, the function will return None.

        .. seealso::

            - :meth:`py3dframe.Frame.from_global_to_frame` method to convert a point or vector from global coordinates to frame coordinates.
            - :meth:`py3dframe.Frame.from_frame_to_parent` method to convert a point or vector from frame coordinates to parent frame coordinates.

        Parameters
        ----------
        point : numpy.ndarray, optional
            Point in frame coordinates with shape (3, N), by default None.

        vector : numpy.ndarray, optional
            Vector in frame coordinates with shape (3, N), by default None.
        
        transpose : bool, optional
            Set if the input points or vectors are given in (N, 3) shape. The default is False.
        
        Returns
        -------
        numpy.ndarray
            Point or vector in global coordinates with shape (3, N), (or (N, 3) if transpose is True).

        Examples
        --------

        >>> parent = Frame(origin=numpy.array([1.0, -1.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True, parent=parent)
        >>> point_frame = numpy.array([[1.0], [2.0], [3.0]])
        >>> point_global = frame.from_frame_to_global(point=point_frame)

        """
        global_frame = self.get_global_frame()
        return global_frame.from_frame_to_parent(point=point, vector=vector, transpose=transpose)


    
    def compose(self, other: Frame) -> Frame:
        r"""
        Compose the current frame with another frame.

        The current frame is define from the global to the frame (1).

        .. math::

            \mathbf{p}_{\text{global}} = \mathbf{R_{(1)}} \mathbf{p}_{(1)} + \mathbf{O}_{(1)}

        The other frame is define from the frame (1) to the frame (2).

        .. math::

            \mathbf{p}_{(1)} = \mathbf{R_{(2)}} \mathbf{p}_{(2)} + \mathbf{O}_{(2)}

        The composition is define from the global to the frame (2).

        .. math::

            \mathbf{p}_{\text{global}} = \mathbf{R_{(1)}} \mathbf{R_{(2)}} \mathbf{p}_{(2)} + \mathbf{R_{(1)}} \mathbf{O}_{(2)} + \mathbf{O}_{(1)}

        .. seealso::

            Using the multiplication operator is equivalent to this method.

            .. code-block:: python

                frame_w_2 = frame_w_1 * frame_1_2

        Parameters
        ----------
        other : Frame
            The other Frame object to compose.
        
        Returns
        -------
        Frame
            The composition of the two frames.
        
        Examples
        --------

        >>> frame_global_to_1 = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame_1_to_2 = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame_global_to_2 = frame1.compose(frame2)

        """
        new_origin = self.origin + numpy.dot(self.rotation_matrix, other.origin)
        new_rotation_matrix = numpy.dot(self.rotation_matrix, other.rotation_matrix)
        return Frame(origin=new_origin, rotation_matrix=new_rotation_matrix, O3_project=True, parent=self.parent)
    


    # Overridden methods
    def __repr__(self) -> str:
        """ 
        String representation of the Frame object. 
        """
        return f"Frame(origin={self.origin}, quaternion={self.quaternion}, direct={self.direct}, has parent={self.parent is not None})"



    def __eq__(self, other: Frame) -> bool:
        """
        Check if two Frame objects are equal. 

        To be equal, the global frame of the two frames must have the same origin, quaternion, and direct properties.
        """
        if not isinstance(other, Frame):
            return False
        
        # Check the origin, quaternion, and direct properties of the global frame
        global_frame = self.get_global_frame()
        other_global_frame = other.get_global_frame()

        same_origin = numpy.allclose(global_frame.origin, other_global_frame.origin, atol=self._tolerance)
        same_quaternion = numpy.allclose(global_frame.quaternion, other_global_frame.quaternion, atol=self._tolerance)
        same_direct = global_frame.direct == other_global_frame.direct

        return same_origin and same_quaternion and same_direct
    


    def __ne__(self, other: Frame) -> bool:
        """ Check if two Frame objects are not equal. """
        return not self.__eq__(other)
    


    def __mul__(self, other: Frame) -> Frame:
        """ 
        Multiply two Frame objects. Return the composition of the two frames.

        The self frame is define from the global to the frame (1).
        The other frame is define from the frame (1) to the frame (2).

        The composition is define from the global to the frame (2).

        .. seealso::

            :meth:`py3dframe.Frame.compose` method to compose two frames.

        Parameters
        ----------
        other : Frame
            The other Frame object to multiply.

        Returns
        -------
        Frame
            The composition of the two frames.
        """
        return self.compose(other)

    

    def __pow__(self, power: int) -> Frame:
        r"""
        Raise the Frame object to a power.

        If the power is 0, the identity frame is returned.
        If the power is 1, the Frame object is returned.
        Otherwise, the Frame object is multiplied by itself power times.

        Parameters
        ----------
        power : int
            The power to raise the Frame object.
        
        Returns
        -------
        Frame
            The Frame object raised to the power.

        Raises
        ------
        ValueError
            If the power is not an integer or if the power is negative.
        """
        if not isinstance(power, int):
            raise TypeError("The power must be an integer.")
        if power < 0:
            raise ValueError("The power must be greater or equal to -1.")
        if power == 0:
            return Frame(origin=numpy.zeros((3,1)), quaternion=numpy.array([1.0, 0.0, 0.0, 0.0]), direct=True)
        elif power == 1:
            return self
        else:
            return self.__mul__(self.__pow__(power - 1))



    def save_to_dict(self, description: str = "", parent: bool = True) -> Dict:
        r"""
        Export the Frame's data to a dictionary.

        The structure of the dictionary is as follows:

        .. code-block:: python

            {
                "type": "Frame [py3dframe]",
                "description": "Description of the frame",
                "origin": [1.0, 2.0, 3.0],
                "quaternion": [0.5, 0.5, 0.5, 0.5],
                "direct": True
                "parent": {
                    "type": "Frame [py3dframe]",
                    "description": "Description of the parent frame",
                    "origin": [1.0, -1.0, 3.0],
                    "quaternion": [0.5, 0.5, 0.5, 0.5],
                    "direct": True
                }
            }

        Parameters
        ----------
        description : str, optional
            A description of the frame, by default "".

        parent : bool, optional
            Set if the parent frame is included in the dictionary, by default True.

        Returns
        -------
        dict
            A dictionary containing the frame's data.

        Raises
        ------
        ValueError
            If the description is not a string.
            If the parent is not a boolean.
        """
        # Check the description
        if not isinstance(description, str):
            raise ValueError("Description must be a string.")
        
        # Check the no_parent
        if not isinstance(parent, bool):
            raise ValueError("parent must be a boolean.")
        
        # Create the dictionary
        data = {
            "type": "Frame [py3dframe]",
            "origin": self.origin.tolist(),
            "quaternion": self.quaternion.tolist(),
            "direct": self.direct
        }

        # Add the description
        if len(description) > 0:
            data["description"] = description

        # Add the parent frame
        if self.parent is not None and parent:
            data["parent"] = self.parent.save_to_dict()
        
        return data



    @classmethod
    def load_from_dict(cls, data: Dict, parent: bool = True) -> Frame:
        r"""
        Create a Frame instance from a dictionary.

        The structure of the dictionary should be as provided by the :meth:`py3dframe.Frame.save_to_dict` method.

        Where the quaternion is given is the scalar-first convention.

        If origin is not given, the default value is None - [0.0, 0.0, 0.0]
        If quaternion is not given, the default value is None - [1.0, 0.0, 0.0, 0.0]
        If direct is not given, the default value is True.
        If parent is not given, the default value is None.

        The other keys are ignored.

        Parameters
        ----------
        data : dict
            A dictionary containing the frame's data.

        parent : bool, optional
            Set if the parent frame is included in the dictionary, by default True.
        
        Returns
        -------
        Frame
            A Frame instance.
        
        Raises
        ------
        ValueError
            If the data is not a dictionary.
            If parent is not a boolean.
        """
        # Check for the input type
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary.")
        
        if not isinstance(parent, bool):
            raise ValueError("parent must be a boolean.")
        
        # Create the Frame instance
        if "origin" in data.keys():
            origin = numpy.array(data["origin"]).reshape((3,1)).astype(float)
        else:
            origin = numpy.zeros((3,1)).astype(float)
        
        if "quaternion" in data.keys():
            quaternion = numpy.array(data["quaternion"]).reshape((4,)).astype(float)
        else:
            quaternion = numpy.array([1.0, 0.0, 0.0, 0.0]).astype(float)
        
        if "direct" in data.keys():
            direct = data["direct"]
        else:
            direct = True

        if "parent" in data.keys() and parent:
            parent_frame = cls.load_from_dict(data["parent"])
        else:
            parent_frame = None
        
        return Frame(origin=origin, quaternion=quaternion, direct=direct, parent=parent_frame)



    def save_to_json(self, filepath: str, description: str = "", parent: bool = True) -> None:
        """
        Export the Frame's data to a JSON file.

        The structure of the JSON file follows the :meth:`py3dframe.Frame.save_to_dict` method.

        Parameters
        ----------
        filepath : str
            The path to the JSON file.
        
        description : str, optional
            A description of the frame, by default "".

        parent : bool, optional
            Set if the parent frame is included in the dictionary, by default True.
        
        Raises
        ------
        FileNotFoundError
            If the filepath is not a valid path.
        """
        # Create the dictionary
        data = self.save_to_dict(description=description, parent=parent)

        # Save the dictionary to a JSON file
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)


    
    @classmethod
    def load_from_json(cls, filepath: str, parent: bool = True) -> Frame:
        """
        Create a Frame instance from a JSON file.

        The structure of the JSON file follows the :meth:`py3dframe.Frame.save_to_dict` method.

        Parameters
        ----------
        filepath : str
            The path to the JSON file.
        
        parent : bool, optional
            Set if the parent frame is included in the dictionary, by default True.
        
        Returns
        -------
        Frame
            A Frame instance.
        
        Raises
        ------
        FileNotFoundError
            If the filepath is not a valid path.
        """
        # Load the dictionary from the JSON file
        with open(filepath, "r") as file:
            data = json.load(file)
        
        # Create the Frame instance
        return cls.load_from_dict(data, parent=parent)