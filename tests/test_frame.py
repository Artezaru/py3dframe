from py3dframe import Frame
import unittest
import numpy as np


class TestFrame(unittest.TestCase):
    def test_default_initialization(self):
        """Test the default initialization of a Frame."""
        f = Frame()
        self.assertTrue(
            np.allclose(f.origin, np.zeros((3, 1)), atol=1e-6),
            "Default origin should be a 3x1 zero vector."
        )
        expected_quat = np.array([1, 0, 0, 0], dtype=np.float32)
        self.assertTrue(
            np.allclose(f.quaternion, expected_quat, atol=1e-6),
            "Default quaternion should be [1, 0, 0, 0]."
        )

    def test_quaternion_initialization(self):
        """Test initialization with a quaternion (90° rotation about the X-axis)."""
        # Quaternion for 90° rotation around X (scalar-first convention)
        quat = np.array([0.7071, 0.7071, 0, 0], dtype=np.float32)
        origin = np.array([1, 2, 3]).reshape((3, 1))
        f = Frame(origin=origin, quaternion=quat, direct=True)
        self.assertTrue(
            np.allclose(f.origin, origin, atol=1e-6),
            "Origin is not set correctly."
        )
        # Expected rotation matrix for 90° rotation about X:
        # [ [1, 0, 0],
        #   [0, 0, -1],
        #   [0, 1, 0] ]
        expected_R = np.array([[1, 0, 0],
                               [0, 0, -1],
                               [0, 1, 0]], dtype=np.float32)
        self.assertTrue(
            np.allclose(f.rotation_matrix, expected_R, atol=1e-2),
            "Rotation matrix does not match the expected 90° rotation around the X-axis."
        )

    def test_rotation_matrix_initialization(self):
        """Test initialization with a rotation matrix (90° rotation about the Z-axis)."""
        theta = np.pi / 2
        Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta),  np.cos(theta), 0],
                       [0,              0,             1]], dtype=np.float32)
        origin = np.zeros((3, 1), dtype=np.float32)
        f = Frame(origin=origin, rotation_matrix=Rz)
        self.assertTrue(
            np.allclose(f.rotation_matrix, Rz, atol=1e-6),
            "Initialization via rotation_matrix failed."
        )

    def test_world_to_local_correct(self):
        """Test converting a point from world to local coordinates."""
        theta = np.pi / 2
        Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta),  np.cos(theta), 0],
                       [0,              0,             1]], dtype=np.float32)
        origin = np.array([1, 1, 1]).reshape((3, 1))
        f = Frame(origin=origin, rotation_matrix=Rz)
        point_world = np.array([[2], [1], [0]], dtype=np.float32)
        point_local = f.from_world_to_local(point=point_world)
        expected_local = np.array([[0], [-1], [-1]], dtype=np.float32)
        self.assertTrue(
            np.allclose(point_local, expected_local, atol=1e-6),
            "World -> local conversion did not produce the expected result."
        )

    def test_world_to_local_conversion(self):
        """Test converting a point from world to local coordinates and back."""
        quat = np.array([0.7071, 0.7071, 0, 0], dtype=np.float32)
        origin = np.array([1, 2, 3]).reshape((3, 1))
        f = Frame(origin=origin, quaternion=quat)
        point_world = np.array([[4], [5], [6]], dtype=np.float32)
        point_local = f.from_world_to_local(point=point_world)
        point_world_reconstructed = f.from_local_to_world(point=point_local)
        self.assertTrue(
            np.allclose(point_world, point_world_reconstructed, atol=1e-4),
            "World <-> local conversion did not properly invert the transformation."
        )

    def test_inverse_frame(self):
        """Test the generation and application of the inverse frame."""
        quat = np.array([0.7071, 0.7071, 0, 0], dtype=np.float32)
        origin = np.array([1, 2, 3]).reshape((3, 1))
        f = Frame(origin=origin, quaternion=quat)
        inv_f = f.get_inverse_frame()
        point_world = np.array([[7], [8], [9]], dtype=np.float32)
        point_local = f.from_world_to_local(point=point_world)
        point_world_from_inv = inv_f.from_world_to_local(point=point_local)
        self.assertTrue(
            np.allclose(point_world, point_world_from_inv, atol=1e-4),
            "Inverse frame transformation did not return the original world point."
        )

    def test_compose_frame(self):
        """Test composing two frames."""
        # Frame 1: 90° rotation about Z-axis.
        theta = np.pi / 2
        Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta),  np.cos(theta), 0],
                       [0,              0,             1]], dtype=np.float32)
        origin1 = np.array([1, 1, 1]).reshape((3, 1))
        f1 = Frame(origin=origin1, rotation_matrix=Rz)
        # Frame 2: 90° rotation about X-axis.
        quat = np.array([0.7071, 0.7071, 0, 0], dtype=np.float32)
        origin2 = np.array([2, 2, 2]).reshape((3, 1))
        f2 = Frame(origin=origin2, quaternion=quat)
        # Compute the point from world -> 1 -> 2.
        point_world = np.array([[2], [1], [-1]], dtype=np.float32)
        point_local_1 = f1.from_world_to_local(point=point_world)
        point_local_2 = f2.from_world_to_local(point=point_local_1)
        composed = f1.compose(f2)
        point_local_composed = composed.from_world_to_local(point=point_world)
        self.assertTrue(
            np.allclose(point_local_2, point_local_composed, atol=1e-6),
            "Composed frame did not produce the expected local point."
        )

    def test_dump_load(self):
        """Test dumping a Frame to a dictionary and loading it back."""
        quat = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        origin = np.array([1, 2, 3]).reshape((3, 1))
        f = Frame(origin=origin, quaternion=quat, direct=False)
        data = f.dump()
        f_loaded = Frame.load(data)
        self.assertEqual(
            f, f_loaded,
            "Dump/Load did not produce an identical Frame."
        )

if __name__ == '__main__':
    unittest.main()
