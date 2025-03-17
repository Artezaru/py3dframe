from __future__ import annotations
from typing import Optional, Dict
import numpy
import json
from scipy.spatial.transform import Rotation

class Frame(object):
    r"""
    Represents a orthonormal reference frame in :math:`\mathbb{R}^3`.

    The frame is defined by a origin and an orientation.
    The orientation can be defined using a rotation matrix.
    The frame can be direct or indirect, depending on the determinant of the rotation matrix.
    The convention for quaternion is scalar first :math:`[w, x, y, z]`.

    The frame use scipy.spatial.transform.Rotation to manage the rotation matrix.

    By convention, the frame is defined from the global frame to the frame frame. 
    Thats means, the coordinates of the rotation matrix are the coordinates of the frame axis in the global coordinates.

    Lets consider a frame :math:`\mathcal{F}` defined by 3 vectors :math:`\mathbf{i}`, :math:`\mathbf{j}`, :math:`\mathbf{k}`.
    The rotation matrix :math:`\mathbf{R}` of the frame is defined by the following equation:

    .. math::

        \mathbf{R} = \begin{bmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \end{bmatrix}

    When the rotation matrix is set, the determinant is checked to determine if the frame is direct or indirect.

    However when the user create the frame using the quaternion or the euler angles, the frame is always considered as direct.
    Then the user must set the frame as indirect using the set_indirect method.
    In this case the rotation matrix is inverted to have a right-handed frame.
    The associated direct frame correspond to the frame with the same origin but the frame Z-axis inverted.

    In fact when the frame is indirect, the user can : 
    - set directly the indirect rotation matrix.
    - set the quaternion or the euler angles of the associated direct frame and then set the frame as indirect.

    The associated direct frame correspond to the frame with the same origin but the frame Z-axis inverted.

    .. seealso:: 
    
        :class:`py3dframe.FrameTree` To manage multiple frames and their relationships.

    Parameters
    ----------
    origin : numpy.ndarray, optional
        The origin of the frame in 3D space with shape (3,1). The default is None - [0.0, 0.0, 0.0].
    
    quaternion : numpy.ndarray, optional
        The quaternion of the frame with shape (4,). The default is None - [1.0, 0.0, 0.0, 0.0].
    
    rotation_matrix : numpy.ndarray, optional
        The rotation matrix of the frame with shape (3,3). The default is None.

    O3_project : bool, optional
        Set if the rotation matrix is projected to the orthogonal group O(3). The default is False.
    
    euler_angles : numpy.ndarray, optional
        The Euler angles of the frame in radians with shape (3,). The default is None.
    
    rotation_vector : numpy.ndarray, optional 
        The rotation vector of the frame with shape (3,). The default is None.
    
    direct : bool, optional
        Set if the frame is direct or indirect. The default is True.


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
        Project a matrix to the orthogonal group O(3) using SVD and minimisation of the frobenius norm.

        The orthogonal group O(3) is the set of 3x3 matrices with determinant 1.

        To project a matrix to O(3), the SVD is computed and the orthogonal matrix is obtained by:

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
            The O(3) projection of the matrix.

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



    def _symmetric_conversion(self, point: numpy.ndarray, normal: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
        r"""
        Convert the frame to the symmetric frame with respect to a plane.

        The plane is defined by a point :math:`\mathbf{p}` and a normal vector :math:`\mathbf{n}`.
        The point and normal vector are given in the global frame coordinates.

        The point and the normal must be arrays with 3 elements.

        For a global point :math:`\mathbf{p}_{\text{global}}`, the symmetric point :math:`\mathbf{p}_{\text{symmetric}}` with respect to the plane is:

        .. math::

            \mathbf{p}_{\text{symmetric}} = \mathbf{p} - 2 <\mathbf{p} - \mathbf{p}_{\text{global}} , \mathbf{n}> \mathbf{n}

        Where :math:`<\cdot, \cdot>` is the dot product.

        Parameters
        ----------
        point : numpy.ndarray
            A point on the plane with shape (3,1).
        
        normal : numpy.ndarray
            The normal vector of the plane with shape (3,1).
        
        Returns
        -------
        numpy.ndarray
            The origin of the symmetric frame.
        
        numpy.ndarray
            The rotation matrix of the symmetric frame.
        
        Raises
        ------
        TypeError
            If the point or the normal is not a numpy array.
        ValueError
            If the point or the normal is not 3x1.
            If the normal vector has zero magnitude.
        """
        # Compute the unit normal vector
        point = numpy.array(point).reshape((3,1)).astype(float)
        normal = numpy.array(normal).reshape((3,1)).astype(float)

        # Compute the unit normal vector
        norm = numpy.linalg.norm(normal, ord=None)
        if abs(norm) < self._tolerance:
            raise ValueError("Normal vector cannot have zero magnitude.")
        normal = normal / norm

        # Compute the reflected origin
        symmetric_origin = self.origin - 2 * numpy.dot(normal.T, self.origin - point) * normal

        # Compute the reflected rotation matrix
        x_axis = self.x_axis - 2 * numpy.dot(normal.T, self.x_axis) * normal
        y_axis = self.y_axis - 2 * numpy.dot(normal.T, self.y_axis) * normal
        z_axis = self.z_axis - 2 * numpy.dot(normal.T, self.z_axis) * normal
        symmetric_rotation_matrix = numpy.column_stack((x_axis, y_axis, z_axis))
        return symmetric_origin, symmetric_rotation_matrix



    def _inverse_conversion(self) -> (numpy.ndarray, numpy.ndarray):
        r"""
        By convention, the frame is defined from the global to the frame.
        This method compute the frame defined from the frame to the global frame.

        By convention, the frame is defined from the global to the frame.

        .. math::

            \mathbf{p}_{\text{frame}} = \mathbf{R}^T (\mathbf{p}_{\text{global}} - \mathbf{O})
        
        So the inverse frame is defined by:

        .. math::

            \mathbf{R}_{\text{inverse}} = \mathbf{R}^T

        .. math::

            \mathbf{p}_{\text{inverse}} = -\mathbf{R}^T \mathbf{O}

        Returns
        -------
        numpy.ndarray
            The origin of the inverse frame.
        
        numpy.ndarray
            The rotation matrix of the inverse frame.    
        """
        # Compute the rotation matrix
        inverse_rotation_matrix = self.rotation_matrix.T
        inverse_origin = -numpy.dot(inverse_rotation_matrix, self.origin)
        return inverse_origin, inverse_rotation_matrix



    # Public methods
    def set_direct(self, direct: bool = True) -> None:
        r"""
        Set the frame as direct or indirect.

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

        Parameters
        ----------
        matrix : numpy.ndarray
           The :math:`O(3)` rotation matrix with shape (3,3).
        
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



    def from_global_to_frame(
        self, 
        *, 
        point: Optional[numpy.ndarray] = None, 
        vector: Optional[numpy.ndarray] = None,
        transpose: bool = False
        ) -> Optional[numpy.ndarray]:
        r"""
        Convert a point or vector from global coordinates to frame coordinates.

        For a point, the equation is:

        .. math::

            \mathbf{p}_{\text{frame}} = \mathbf{R}^T (\mathbf{p}_{\text{global}} - \mathbf{O})

        For a vector, the equation is:

        .. math::

            \mathbf{v}_{\text{frame}} = \mathbf{R}^T \mathbf{v}_{\text{global}}

        if point is not None and vector is not None, the function will raise a ValueError.
        if point is None and vector is None, the function will return None.

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
            Point or vector in frame coordinates with shape (3, N).
        
        Examples
        --------
        >>> frame = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame.from_global_to_frame(point=numpy.array([[1.0], [2.0], [3.0]]))
        """
        if point is not None and vector is not None:
            raise ValueError("Only one of 'point' or 'vector' can be provided.")
        if point is None and vector is None:
            return None

        data = point if point is not None else vector
        
        # Convert the data into (3, N) shape
        if transpose:
            data = numpy.array(data).reshape((-1, 3)).astype(float).T
        else:
            data = numpy.array(data).reshape((3, -1)).astype(float)

        # Convert the point to vector
        if point is not None:
            data = data - self.origin
        
        # Convert the data to frame coordinates
        frame_data = numpy.dot(self.rotation_matrix.T, data)

        # Organize the output array
        if transpose:
            frame_data = frame_data.T
        return frame_data



    def from_frame_to_global(
        self, 
        *, 
        point: Optional[numpy.ndarray] = None,
        vector: Optional[numpy.ndarray] = None,
        transpose: bool = False
        ) -> Optional[numpy.ndarray]:
        r"""
        Convert a point or vector from frame coordinates to global coordinates.

        For a point, the equation is:

        .. math::

            \mathbf{p}_{\text{global}} = \mathbf{R} \mathbf{p}_{\text{frame}} + \mathbf{O}

        For a vector, the equation is:

        .. math::   
    
            \mathbf{v}_{\text{global}} = \mathbf{R} \mathbf{v}_{\text{frame}}

        if point is not None and vector is not None, the function will raise a ValueError.
        if point is None and vector is None, the function will return None.

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
            Point or vector in global coordinates with shape (3, N).
        
        Examples
        --------
        >>> frame = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame.from_global_to_frame(point=numpy.array([[1.0], [2.0], [3.0]]))
        """
        if point is not None and vector is not None:
            raise ValueError("Only one of 'point' or 'vector' can be provided.")
        if point is None and vector is None:
            return None
        
        data = point if point is not None else vector

        # Convert the data into (3, N) shape
        if transpose:
            data = numpy.array(data).reshape((-1, 3)).astype(float).T
        else:
            data = numpy.array(data).reshape((3, -1)).astype(float)
        
        # Convert the data to global coordinates
        global_data = numpy.dot(self.rotation_matrix, data)

        # Convert the vector to point
        if point is not None:
            global_data = global_data + self.origin
        
        # Organize the output array
        if transpose:
            global_data = global_data.T
        return global_data



    def get_symmetric_frame(
        self, 
        point: numpy.ndarray, 
        normal: numpy.ndarray,
        ) -> Frame:
        r"""
        Get the symmetric frame of the current frame with respect to a plane defined by a point and a normal vector.

        The point and the normal must be arrays with 3 elements.

        Parameters
        ----------
        point : numpy.ndarray
            A point on the plane with shape (3,1).
        
        normal : numpy.ndarray
            The normal vector of the plane with shape (3,1).
        
        Returns
        -------
        Frame
            The symmetric frame of the current frame with respect to the plane.
        
        Examples
        --------
        >>> frame = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> point = numpy.array([1.0, 2.0, 3.0])
        >>> normal = numpy.array([0.0, 0.0, 1.0])
        >>> symmetric_frame = frame.get_symmetric_frame(point, normal)
        """
        symmetric_origin, symmetric_rotation_matrix = self._symmetric_conversion(point, normal)
        return Frame(origin=symmetric_origin, rotation_matrix=symmetric_rotation_matrix, O3_project=True)
    


    def apply_symmetric_frame(
        self, 
        point: numpy.ndarray, 
        normal: numpy.ndarray
        ) -> None:
        r"""
        Apply a symmetry to the frame with respect to a plane defined by a point and a normal vector.

        The point and the normal must be arrays with 3 elements.

        Parameters
        ----------
        point : numpy.ndarray
            A point on the plane with shape (3,1).
        
        normal : numpy.ndarray
            The normal vector of the plane with shape (3,1).
        
        Examples
        --------
        >>> frame = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> point = numpy.array([1.0, 2.0, 3.0])
        >>> normal = numpy.array([0.0, 0.0, 1.0])
        >>> frame.apply_symmetric_frame(point, normal)
        """
        symmetric_origin, symmetric_rotation_matrix = self._symmetric_conversion(point, normal)
        self.set_O3_rotation_matrix(symmetric_rotation_matrix)
        self.origin = symmetric_origin



    def get_inverse_frame(self) -> Frame:
        r"""
        Get the inverse frame of the current frame.

        By convention, the frame is defined from the global to the frame.
        This method compute the frame defined from the frame to the global frame.

        Returns
        -------
        Frame
            The inverse frame of the current frame.
        
        Examples
        --------
        >>> frame = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> inverse_frame = frame.get_inverse_frame()
        """
        inverse_origin, inverse_rotation_matrix = self._inverse_conversion()
        return Frame(origin=inverse_origin, rotation_matrix=inverse_rotation_matrix, O3_project=True)



    def apply_inverse_frame(self) -> None:
        r"""
        Apply a inversion to the frame.
        
        By convention, the frame is defined from the global to the frame.
        This method compute the frame defined from the frame to the global frame.

        Examples
        --------
        >>> frame = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame.apply_inverse_frame()
        """
        inverse_origin, inverse_rotation_matrix = self._inverse_conversion()
        self.set_O3_rotation_matrix(inverse_rotation_matrix)
        self.origin = inverse_origin


    
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
        >>> frame_w_1 = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame_1_2 = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame_w_2 = frame1.compose(frame2)
        """
        return self * other
    


    # Overridden methods
    def __repr__(self) -> str:
        """ String representation of the Frame object. """
        return f"Frame(origin={self.origin}, quaternion={self.quaternion}, direct={self.direct})"

    def __eq__(self, other: Frame) -> bool:
        """ Check if two Frame objects are equal. """
        return numpy.allclose(self.origin, other.origin, atol=self._tolerance) and numpy.allclose(self.quaternion, other.quaternion, atol=self._tolerance) and self.direct == other.direct

    def __ne__(self, other: Frame) -> bool:
        """ Check if two Frame objects are not equal. """
        return not self.__eq__(other)
    
    def __mul__(self, other: Frame) -> Frame:
        """ 
        Multiply two Frame objects. Return the composition of the two frames.

        The self frame is define from the global to the frame (1).
        The other frame is define from the frame (1) to the frame (2).

        The composition is define from the global to the frame (2).

        Parameters
        ----------
        other : Frame
            The other Frame object to multiply.

        Returns
        -------
        Frame
            The composition of the two frames.
        """
        new_origin = self.origin + numpy.dot(self.rotation_matrix, other.origin)
        new_rotation_matrix = numpy.dot(self.rotation_matrix, other.rotation_matrix)
        return Frame(origin=new_origin, rotation_matrix=new_rotation_matrix, O3_project=True)    

    def __pow__(self, power: int) -> Frame:
        r"""
        Raise the Frame object to a power.

        If the power is 0, the identity frame is returned.
        If the power is 1, the Frame object is returned.
        If the power is -1, the inverse Frame object is returned.
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
        if power < -1:
            raise ValueError("The power must be greater or equal to -1.")
        if power == 0:
            return Frame(origin=numpy.zeros((3,1)), quaternion=numpy.array([1.0, 0.0, 0.0, 0.0]), direct=True)
        elif power == 1:
            return self
        elif power == -1:
            return self.get_inverse_frame()
        else:
            return self.__mul__(self.__pow__(power - 1))



    def save_to_dict(self, description: str = "") -> Dict:
        """
        Export the Frame's data to a dictionary.

        The structure of the dictionary is as follows:

        .. code-block:: python

            {
                "type": "Frame [py3dframe]",
                "description": "Description of the frame",
                "origin": [1.0, 2.0, 3.0],
                "quaternion": [0.5, 0.5, 0.5, 0.5],
                "direct": True
            }

        Parameters
        ----------
        description : str, optional
            A description of the frame, by default "".

        Returns
        -------
        dict
            A dictionary containing the origin, quaternion, and direct.

        Raises
        ------
        ValueError
            If the description is not a string.
        """
        # Check the description
        if not isinstance(description, str):
            raise ValueError("Description must be a string.")
        
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
        
        return data



    @classmethod
    def load_from_dict(cls, data: Dict) -> Frame:
        """
        Create a Frame instance from a dictionary.

        The structure of the dictionary should be as provided by the :meth:`py3dframe.Frame.save_to_dict` method.

        Where the quaternion is given is the scalar-first convention.

        If origin is not given, the default value is None - [0.0, 0.0, 0.0]
        If quaternion is not given, the default value is None - [1.0, 0.0, 0.0, 0.0]
        If direct is not given, the default value is True.

        The other keys are ignored.

        Parameters
        ----------
        data : dict
            A dictionary containing the origin, quaternion, and direct.
        
        Returns
        -------
        Frame
            A Frame instance.
        
        Raises
        ------
        ValueError
            If the data is not a dictionary.
        """
        # Check for the input type
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary.")
        
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
        
        return Frame(origin=origin, quaternion=quaternion, direct=direct)



    def save_to_json(self, filepath: str, description: str = "") -> None:
        """
        Export the Frame's data to a JSON file.

        The structure of the JSON file follows the :meth:`py3dframe.Frame.save_to_dict` method.

        Parameters
        ----------
        filepath : str
            The path to the JSON file.
        
        description : str, optional
            A description of the frame, by default "".
        
        Raises
        ------
        FileNotFoundError
            If the filepath is not a valid path.
        """
        # Create the dictionary
        data = self.save_to_dict(description=description)

        # Save the dictionary to a JSON file
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)


    
    @classmethod
    def load_from_json(cls, filepath: str) -> Frame:
        """
        Create a Frame instance from a JSON file.

        The structure of the JSON file follows the :meth:`py3dframe.Frame.save_to_dict` method.

        Parameters
        ----------
        filepath : str
            The path to the JSON file.
        
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
        return cls.load_from_dict(data)