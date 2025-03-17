from py3dframe import Frame
import numpy as np
import pytest

def test_default_initialization():
    """Test the default initialization of a Frame."""
    f = Frame()
    assert np.allclose(f.origin, np.zeros((3, 1)), atol=1e-6), \
        "Default origin should be a 3x1 zero vector."
    
    expected_quat = np.array([1, 0, 0, 0], dtype=np.float64)
    assert np.allclose(f.quaternion, expected_quat, atol=1e-6), \
        "Default quaternion should be [1, 0, 0, 0]."

def test_quaternion_initialization():
    """Test initialization with a quaternion (90° rotation about the X-axis)."""
    quat = np.array([0.7071, 0.7071, 0, 0], dtype=np.float64)
    origin = np.array([1, 2, 3]).reshape((3, 1))
    f = Frame(origin=origin, quaternion=quat, direct=True)

    assert np.allclose(f.origin, origin, atol=1e-6), "Origin is not set correctly."

    expected_R = np.array([[1, 0, 0],
                           [0, 0, -1],
                           [0, 1, 0]], dtype=np.float64)
    assert np.allclose(f.rotation_matrix, expected_R, atol=1e-2), \
        "Rotation matrix does not match the expected 90° rotation around the X-axis."

def test_rotation_matrix_initialization():
    """Test initialization with a rotation matrix (90° rotation about the Z-axis)."""
    theta = np.pi / 2
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta),  np.cos(theta), 0],
                   [0,              0,             1]], dtype=np.float64)
    origin = np.zeros((3, 1), dtype=np.float64)
    f = Frame(origin=origin, rotation_matrix=Rz)

    assert np.allclose(f.rotation_matrix, Rz, atol=1e-6), \
        "Initialization via rotation_matrix failed."

def test_global_to_frame_correct():
    """Test converting a point from global to frame coordinates."""
    theta = np.pi / 2
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta),  np.cos(theta), 0],
                   [0,              0,             1]], dtype=np.float64)
    origin = np.array([1, 1, 1]).reshape((3, 1))
    f = Frame(origin=origin, rotation_matrix=Rz)

    point_global = np.array([[2], [1], [0]], dtype=np.float64)
    point_frame = f.from_global_to_frame(point=point_global)
    expected_frame = np.array([[0], [-1], [-1]], dtype=np.float64)

    assert np.allclose(point_frame, expected_frame, atol=1e-6), \
        "global -> frame conversion did not produce the expected result."

def test_global_to_frame_transpose_correct():
    """Test converting a point from global to frame coordinates."""
    theta = np.pi / 2
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta),  np.cos(theta), 0],
                   [0,              0,             1]], dtype=np.float64)
    origin = np.array([1, 1, 1]).reshape((3, 1))
    f = Frame(origin=origin, rotation_matrix=Rz)

    point_global = np.array([[2], [1], [0]], dtype=np.float64).T
    point_frame = f.from_global_to_frame(point=point_global, transpose=True)
    expected_frame = np.array([[0], [-1], [-1]], dtype=np.float64).T

    assert np.allclose(point_frame, expected_frame, atol=1e-6), \
        "global -> frame conversion did not produce the expected result."

def test_global_to_frame_conversion():
    """Test converting a point from global to frame coordinates and back."""
    quat = np.array([0.7071, 0.7071, 0, 0], dtype=np.float64)
    origin = np.array([1, 2, 3]).reshape((3, 1))
    f = Frame(origin=origin, quaternion=quat)

    point_global = np.array([[4], [5], [6]], dtype=np.float64)
    point_frame = f.from_global_to_frame(point=point_global)
    point_global_reconstructed = f.from_frame_to_global(point=point_frame)

    assert np.allclose(point_global, point_global_reconstructed, atol=1e-4), \
        "global <-> frame conversion did not properly invert the transformation."

def test_inverse_frame():
    """Test the generation and application of the inverse frame."""
    quat = np.array([0.7071, 0.7071, 0, 0], dtype=np.float64)
    origin = np.array([1, 2, 3]).reshape((3, 1))
    f = Frame(origin=origin, quaternion=quat)
    inv_f = f.get_inverse_frame()

    point_global = np.array([[7], [8], [9]], dtype=np.float64)
    point_frame = f.from_global_to_frame(point=point_global)
    point_global_from_inv = inv_f.from_global_to_frame(point=point_frame)

    assert np.allclose(point_global, point_global_from_inv, atol=1e-4), \
        "Inverse frame transformation did not return the original global point."

def test_compose_frame():
    """Test composing two frames."""
    theta = np.pi / 2
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta),  np.cos(theta), 0],
                   [0,              0,             1]], dtype=np.float64)
    origin1 = np.array([1, 1, 1]).reshape((3, 1))
    f1 = Frame(origin=origin1, rotation_matrix=Rz)

    quat = np.array([0.7071, 0.7071, 0, 0], dtype=np.float64)
    origin2 = np.array([2, 2, 2]).reshape((3, 1))
    f2 = Frame(origin=origin2, quaternion=quat)

    point_global = np.array([[2], [1], [-1]], dtype=np.float64)
    point_frame_1 = f1.from_global_to_frame(point=point_global)
    point_frame_2 = f2.from_global_to_frame(point=point_frame_1)

    composed = f1.compose(f2)
    point_frame_composed = composed.from_global_to_frame(point=point_global)

    assert np.allclose(point_frame_2, point_frame_composed, atol=1e-6), \
        "Composed frame did not produce the expected frame point."

def test_save_load_dict():
    """Test sabing a Frame to a dictionary and loading it back."""
    quat = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    origin = np.array([1, 2, 3]).reshape((3, 1))
    f = Frame(origin=origin, quaternion=quat, direct=False)

    data = f.save_to_dict()
    f_loaded = Frame.load_from_dict(data)

    assert np.allclose(f.origin, f_loaded.origin, atol=1e-6), \
        "Loaded origin does not match the original."
    assert np.allclose(f.quaternion, f_loaded.quaternion, atol=1e-6), \
        "Loaded quaternion does not match the original."
    assert f.direct == f_loaded.direct, \
        "Loaded direct flag does not match the original."
    
def test_save_load_json(tmp_path):
    """Test saving a Frame to a JSON file and loading it back."""
    quat = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    origin = np.array([1, 2, 3]).reshape((3, 1))
    f = Frame(origin=origin, quaternion=quat, direct=False)

    filepath = tmp_path / "test_frame.json"

    f.save_to_json(filepath)
    f_loaded = Frame.load_from_json(filepath)

    assert np.allclose(f.origin, f_loaded.origin, atol=1e-6), \
        "Loaded origin does not match the original."
    assert np.allclose(f.quaternion, f_loaded.quaternion, atol=1e-6), \
        "Loaded quaternion does not match the original."
    assert f.direct == f_loaded.direct, \
        "Loaded direct flag does not match the original."


