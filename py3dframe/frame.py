from __future__ import annotations
from typing import Optional, Dict

import numpy
from scipy.spatial.transform import Rotation

class Frame(object):
    r"""
    Represents a orthonormal reference frame in :math:`\mathbb{R}^3`.

    The frame is defined by a origin and an orientation.
    The orientation can be defined using a quaternion, a rotation matrix, or Euler angles.
    The frame can be direct or indirect, depending on the determinant of the rotation matrix.
    The convention for quaternion is scalar first :math:`[w, x, y, z]`.

    By convention, the frame is defined from the world frame to the local frame. 
    Thats means, the coordinates of the rotation matrix are the coordinates of the local axis in the world coordinates.

    Lets consider a frame :math:`\mathcal{F}` defined by 3 vectors :math:`\mathbf{i}`, :math:`\mathbf{j}`, :math:`\mathbf{k}`.
    The rotation matrix :math:`\mathbf{R}` of the frame is defined by the following equation:

    .. math::

        \mathbf{R} = \begin{bmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \end{bmatrix}
    
    When the rotation matrix is set, the determinant is checked to determine if the frame is direct or indirect.

    However when the quaternion or the euleur angles is set, the frame is always considered direct. 
    Then the user must set the frame as indirect using the set_indirect method.
    In this case the rotation matrix is inverted to have a right-handed frame.

    In fact when the frame is indirect, the user can : 
    - set directly the indirect rotation matrix.
    - set the quaternion or the euler angles of the associated direct frame and then set the frame as indirect.

    The associated direct frame correspond to the frame with the same origin but the local Z-axis inverted.

    .. seealso:: :class:`FrameBinder`
        To manage multiple frames and their relationships.


    Properties
    ----------
    origin : numpy.ndarray
        Get or set the origin of the frame in 3D space with shape (3,1). 

    quaternion : numpy.ndarray
        Get or set the quaternion of the frame.

    rotation_matrix : numpy.ndarray
        Get the rotation matrix of the frame.

    euler_angles : numpy.ndarray
        Get the Euler angles of the frame in radians.

    homogeneous_matrix : numpy.ndarray
        Get the homogeneous transformation matrix of the frame.

    direct : bool
        Get or set if the frame is direct or indirect.

    x_axis : numpy.ndarray
        Get the X-axis of the frame with shape (3,1)

    y_axis : numpy.ndarray
        Get the Y-axis of the frame with shape (3,1)

    z_axis : numpy.ndarray
        Get the Z-axis of the frame with shape (3,1)

    is_direct : bool
        Check if the frame is direct.

    is_indirect : bool
        Check if the frame is indirect.

    Examples:
    ---------

    Example of using the `Frame` class to obtain its rotation matrix.

    .. code-block:: python

        frame = Frame(
            origin=numpy.array([1.0, 2.0, 3.0]), 
            quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), 
            direct=True
        )
        print(frame.rotation_matrix)

    Methods:
    --------
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
        direct: bool = True,
        ) -> None:
        # Set default values
        if sum([quaternion is not None, rotation_matrix is not None, euler_angles is not None]) > 1:
            raise ValueError("Only one of 'quaternion', 'rotation_matrix', or 'euler_angles' can be provided.")
        elif sum([quaternion is not None, rotation_matrix is not None, euler_angles is not None]) == 0:
            quaternion = numpy.array([1.0, 0.0, 0.0, 0.0], dtype=numpy.float32)
        if origin is None:
            origin = numpy.zeros((3,1), dtype=numpy.float32)

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



    # Properties getters and setters
    @property
    def origin(self) -> numpy.ndarray:
        """Get or set the origin of the frame in 3D space with shape (3,1)."""
        return self._origin
    
    @origin.setter
    def origin(self, origin: numpy.ndarray) -> None:
        self._origin = numpy.array(origin, dtype=numpy.float32).reshape((3,1))
    


    @property
    def quaternion(self) -> numpy.ndarray:
        """Get or set the quaternion of the frame."""
        quaternion = self._rotation.as_quat(scalar_first=True)
        quaternion = quaternion / numpy.linalg.norm(quaternion, ord=None)
        return quaternion
    
    @quaternion.setter
    def quaternion(self, quaternion: numpy.ndarray) -> None:
        quaternion = numpy.array(quaternion, dtype=numpy.float32).reshape((4,))
        norm = numpy.linalg.norm(quaternion, ord=None)
        if abs(norm) < self._tolerance:
            raise ValueError("Quaternion cannot have zero magnitude.")
        quaternion = quaternion / norm
        self._rotation = Rotation.from_quat(quaternion, scalar_first=True)
    


    @property
    def rotation_matrix(self) -> numpy.ndarray:
        """Get the rotation matrix of the frame."""
        rotation_matrix = self._rotation.as_matrix()
        if not self.direct:
            rotation_matrix[:, 2] = -rotation_matrix[:, 2]
        return rotation_matrix

    @rotation_matrix.setter
    def rotation_matrix(self, rotation_matrix: numpy.ndarray) -> None:
        rotation_matrix = numpy.array(rotation_matrix, dtype=numpy.float32).reshape((3,3))
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
        """Get the Euler angles of the frame in radians."""
        return self._rotation.as_euler("XYZ", degrees=False)

    @euler_angles.setter
    def euler_angles(self, euler_angles: numpy.ndarray) -> None:
        euler_angles = numpy.array(euler_angles, dtype=numpy.float32).reshape((3,))
        self._rotation = Rotation.from_euler("XYZ", euler_angles, degrees=False)
    


    @property
    def homogeneous_matrix(self) -> numpy.ndarray:
        """Get the homogeneous transformation matrix of the frame."""
        matrix = numpy.eye(4)
        matrix[:3, :3] = self.rotation_matrix
        matrix[:3, 3] = self.origin
        return matrix

    @homogeneous_matrix.setter
    def homogeneous_matrix(self, matrix: numpy.ndarray) -> None:
        homogeneous_matrix = numpy.array(matrix, dtype=numpy.float32).reshape((4,4))
        self.origin = homogeneous_matrix[:3, 3]
        self.rotation_matrix = homogeneous_matrix[:3, :3]



    @property
    def direct(self) -> bool:
        """Get or set if the frame is direct or indirect."""
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
        The point and normal vector are given in the world frame coordinates.

        The point and the normal must be arrays with 3 elements.

        For a world point :math:`\mathbf{p}_{\text{world}}`, the symmetric point :math:`\mathbf{p}_{\text{symmetric}}` with respect to the plane is:

        .. math::

            \mathbf{p}_{\text{symmetric}} = \mathbf{p} - 2 <\mathbf{p} - \mathbf{p}_{\text{world}} , \mathbf{n}> \mathbf{n}

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
        point = numpy.array(point, dtype=numpy.float32).reshape((3,1))
        normal = numpy.array(normal, dtype=numpy.float32).reshape((3,1))

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
        By convention, the frame is defined from the world to the local.
        This method compute the frame defined from the local to the world frame.

        For a orthogonal frame :math:`\mathcal{F}` defined by a origin :math:`\mathbf{p}` and a rotation matrix :math:`\mathbf{R}, the inverse frame :math:`\mathcal{F}^{-1}` is defined by:

        .. math::

            \mathbf{p}_{\text{local}} = -\mathbf{R}^T \mathbf{p}_{\text{world}}

        .. math::

            \mathbf{R}_{\text{local}} = \mathbf{R}^T

        Returns
        -------
        numpy.ndarray
            The origin of the inverse frame.
        
        numpy.ndarray
            The rotation matrix of the inverse frame.    
        """
        # Compute the rotation matrix
        inverse_rotation_matrix = self.rotation_matrix.T

        # Compute the origin
        inverse_origin = self.from_world_to_local(point=numpy.array([0.0, 0.0, 0.0]))
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



    def from_world_to_local(
        self, 
        *, 
        point: Optional[numpy.ndarray] = None, 
        vector: Optional[numpy.ndarray] = None
        ) -> Optional[numpy.ndarray]:
        r"""
        Convert a point or vector from world coordinates to local coordinates.

        For a point, the equation is:

        .. math::

            \mathbf{p}_{\text{local}} = \mathbf{R}^T (\mathbf{p}_{\text{world}} - \mathbf{p}_{\text{origin}})

        For a vector, the equation is:

        .. math::

            \mathbf{v}_{\text{local}} = \mathbf{R}^T \mathbf{v}_{\text{world}}

        if point is not None and vector is not None, the function will raise a ValueError.
        if point is None and vector is None, the function will return None.

        Parameters
        ----------
        point : numpy.ndarray, optional
            Point in world coordinates with shape (3, N), by default None.
        
        vector : numpy.ndarray, optional
            Vector in world coordinates with shape (3, N), by default None.
        
        Returns
        -------
        numpy.ndarray
            Point or vector in local coordinates with shape (3, N).
        
        Examples
        --------
        >>> frame = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame.from_world_to_local(point=numpy.array([[1.0], [2.0], [3.0]]))
        """
        if point is not None and vector is not None:
            raise ValueError("Only one of 'point' or 'vector' can be provided.")
        if point is not None:
            point = numpy.array(point, dtype=numpy.float32).reshape((3, -1))
            local_point = numpy.dot(self.rotation_matrix.T, point - self.origin)
            return local_point
        elif vector is not None:
            vector = numpy.array(vector, dtype=numpy.float32).reshape((3, -1))
            local_vector = numpy.dot(self.rotation_matrix.T, vector)
            return local_vector
        else:
            return None



    def from_local_to_world(
        self, 
        *, 
        point: Optional[numpy.ndarray] = None,
        vector: Optional[numpy.ndarray] = None
        ) -> Optional[numpy.ndarray]:
        r"""
        Convert a point or vector from local coordinates to world coordinates.

        For a point, the equation is:

        .. math::

            \mathbf{p}_{\text{world}} = \mathbf{R} \mathbf{p}_{\text{local}} + \mathbf{p}_{\text{origin}}

        For a vector, the equation is:

        .. math::   
    
            \mathbf{v}_{\text{world}} = \mathbf{R} \mathbf{v}_{\text{local}}

        if point is not None and vector is not None, the function will raise a ValueError.
        if point is None and vector is None, the function will return None.

        Parameters
        ----------
        point : numpy.ndarray, optional
            Point in local coordinates with shape (3, N), by default None.
        
        vector : numpy.ndarray, optional
            Vector in local coordinates with shape (3, N), by default None.
        
        Returns
        -------
        numpy.ndarray
            Point or vector in world coordinates with shape (3, N).
        
        Examples
        --------
        >>> frame = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame.from_world_to_local(point=numpy.array([[1.0], [2.0], [3.0]]))
        """
        if point is not None and vector is not None:
            raise ValueError("Only one of 'point' or 'vector' can be provided.")
        if point is not None:
            point = numpy.array(point, dtype=numpy.float32).reshape((3, -1))
            world_point = self.origin + numpy.dot(self.rotation_matrix, point)
            return world_point
        elif vector is not None:
            vector = numpy.array(vector, dtype=numpy.float32).reshape((3, -1))
            world_vector = numpy.dot(self.rotation_matrix, vector)
            return world_vector
        else:
            return None



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

        By convention, the frame is defined from the world to the local.
        This method compute the frame defined from the local to the world frame.

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
        
        By convention, the frame is defined from the world to the local.
        This method compute the frame defined from the local to the world frame.

        Examples
        --------
        >>> frame = Frame(origin=numpy.array([1.0, 2.0, 3.0]), quaternion=numpy.array([0.5, 0.5, 0.5, 0.5]), direct=True)
        >>> frame.apply_inverse_frame()
        """
        inverse_origin, inverse_rotation_matrix = self._inverse_conversion()
        self.set_O3_rotation_matrix(inverse_rotation_matrix)
        self.origin = inverse_origin
    


    # Overridden methods
    def __repr__(self) -> str:
        """ String representation of the Frame object. """
        return f"Frame(origin={self.origin}, quaternion={self.quaternion}, direct={self.direct})"

    def __eq__(self, other: Frame) -> bool:
        """ Check if two Frame objects are equal. """
        return numpy.allclose(self.origin, other.origin, atol=self._tolerance) and numpy.allclose(self.quaternion, other.quaternion, atol=self._tolerance) and self.direct == other.direct



    # Load and dump methods
    def dump(self) -> Dict:
        """
        Export the Frame's data as a dictionary.

        Returns
        -------
        dict
            A dictionary containing the origin, quaternion, and direct.
        """
        return {
            "origin": self.origin.tolist(),
            "quaternion": self.quaternion.tolist(),
            "direct": self.direct,
        }


    @classmethod
    def load(cls, data: Dict) -> Frame:
        """
        Create a Frame instance from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the origin, quaternion, and direct.
        
        Returns
        -------
        Frame
            A Frame instance.
        """
        # Check for required keys
        required_keys = {"origin", "quaternion", "direct"}
        if not required_keys.issubset(data.keys()):
            raise ValueError(f"The dictionary must contain keys: {required_keys}")
        
        # Create the Frame instance
        return cls(
            origin=data["origin"],
            quaternion=data["quaternion"],
            direct=data["direct"],
        )













