import numpy
from scipy.spatial.transform import Rotation

class BasisUtils:
    """
    A utility class for handling transformations between quaternions, basis vectors, 
    and orthonormal basis.
    """

    @staticmethod
    def quaternion_to_basis(quaternion: numpy.ndarray):
        """
        Converts a quaternion to its corresponding basis vectors (X, Y, Z axes).

        Parameters
        ----------
        quaternion : numpy.ndarray
            Quaternion in the form [qw, qx, qy, qz].
        
        Returns
        -------
        tuple
            A tuple of three numpy arrays (X_axis, Y_axis, Z_axis) representing
        """
        if len(quaternion) != 4:
            raise ValueError("Quaternion must have 4 components: [qw, qx, qy, qz].")

        # Normalize the quaternion
        quaternion = numpy.array(quaternion, dtype=float)
        norm = numpy.linalg.norm(quaternion)
        if norm == 0:
            raise ValueError("Quaternion cannot have zero magnitude.")
        quaternion /= norm

        # Convert the quaternion to a rotation matrix
        rotation = Rotation.from_quat(quaternion, scalar_first=True)
        rotation_matrix = rotation.as_matrix()

        # Extract basis vectors from the rotation matrix
        X_axis = rotation_matrix[:, 0] / numpy.linalg.norm(rotation_matrix[:, 0])  # First column
        Y_axis = rotation_matrix[:, 1] / numpy.linalg.norm(rotation_matrix[:, 1])  # Second column
        Z_axis = rotation_matrix[:, 2] / numpy.linalg.norm(rotation_matrix[:, 2])  # Third column

        return X_axis, Y_axis, Z_axis



    @staticmethod
    def basis_to_quaternion(X_axis: numpy.ndarray, Y_axis: numpy.ndarray, Z_axis: numpy.ndarray) -> numpy.ndarray:
        """
        Converts a set of basis vectors (X, Y, Z axes) to a quaternion.

        Parameters
        ----------
        X_axis : numpy.ndarray
            Vector representing the X-axis.

        Y_axis : numpy.ndarray
            Vector representing the Y-axis.

        Z_axis : numpy.ndarray
            Vector representing the Z-axis.

        Returns
        -------
        numpy.ndarray
            Quaternion in the form [qw, qx, qy, qz].
        """
        # Normalize the input vectors
        X_axis = X_axis / numpy.linalg.norm(X_axis)
        Y_axis = Y_axis / numpy.linalg.norm(Y_axis)
        Z_axis = Z_axis / numpy.linalg.norm(Z_axis)

        # Form the rotation matrix
        rotation_matrix = numpy.column_stack((X_axis, Y_axis, Z_axis))

        # Validate the rotation matrix
        if not numpy.allclose(numpy.dot(rotation_matrix.T, rotation_matrix), numpy.eye(3), atol=1e-6):
            raise ValueError("The provided vectors do not form an orthonormal basis.")
        if not numpy.isclose(numpy.linalg.det(rotation_matrix), 1.0):
            raise ValueError("The determinant of the rotation matrix must be 1.")

        # Convert the rotation matrix to a quaternion
        rotation = Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat(scalar_first=True)
        quaternion /= numpy.linalg.norm(quaternion)

        return quaternion



    @staticmethod
    def best_reference_vector(direction: numpy.ndarray) -> numpy.ndarray:
        """
        Selects the best reference vector for building an orthonormal basis
        given a direction vector.

        Parameters
        ----------
        direction : numpy.ndarray
            Direction vector to be used as the Z-axis.
        
        Returns
        -------
        numpy.ndarray
            A reference vector that minimizes its alignment with the given direction.
        """
        # Normalize the input vector
        direction = direction / numpy.linalg.norm(direction)

        # Candidate reference vectors
        candidates = [
            numpy.array([1, 0, 0]), numpy.array([0, 1, 0]), numpy.array([0, 0, 1]),
            numpy.array([1, 1, 0]), numpy.array([1, 0, 1]), numpy.array([0, 1, 1]),
            numpy.array([1, 1, 1])
        ]

        # Normalize candidates
        candidates = [v / numpy.linalg.norm(v) for v in candidates]

        # Compute dot products and select the best candidate
        dot_products = [abs(numpy.dot(direction, v)) for v in candidates]
        best_candidate = candidates[numpy.argmin(dot_products)]

        return best_candidate



    @staticmethod
    def orthonormal_basis_from_direction(direction: numpy.ndarray):
        """
        Computes an orthonormal basis given a direction vector by selecting
        a suitable reference vector.

        Parameters
        ----------
        direction : numpy.ndarray
            Direction vector to be used as the Z-axis.

        Returns
        -------
        tuple
            A tuple (X_axis, Y_axis, Z_axis) representing the orthonormal basis.
        """
        # Normalize the direction vector
        Z_axis = direction / numpy.linalg.norm(direction)

        # Select the best reference vector
        reference = FrameUtils.best_reference_vector(Z_axis)

        # Compute X-axis
        X_axis = numpy.cross(reference, Z_axis)
        X_axis /= numpy.linalg.norm(X_axis)

        # Compute Y-axis
        Y_axis = numpy.cross(Z_axis, X_axis)
        Y_axis /= numpy.linalg.norm(Y_axis)

        return X_axis, Y_axis, Z_axis



    @staticmethod
    def orthonormal_basis_from_plane(direction: numpy.ndarray, plane_vector: numpy.ndarray):
        """
        Computes an orthonormal basis given a direction vector (Z-axis) and a vector in the desired plane.

        Parameters
        ----------
        direction : numpy.ndarray
            Direction vector to be used as the Z-axis.
        
        plane_vector : numpy.ndarray
            Vector in the plane to define the X-axis.

        Returns
        -------
        tuple
            A tuple (X_axis, Y_axis, Z_axis) representing the orthonormal basis.
        """
        # Normalize the direction vector
        Z_axis = direction / numpy.linalg.norm(direction)

        # Project the plane vector onto the plane orthogonal to Z-axis
        X_axis = plane_vector - numpy.dot(Z_axis, plane_vector) * Z_axis
        X_axis /= numpy.linalg.norm(X_axis)

        # Compute Y-axis
        Y_axis = numpy.cross(Z_axis, X_axis)

        return X_axis, Y_axis, Z_axis


        
