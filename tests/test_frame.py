import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from py3dframe import Frame, switch_RT_convention

def test_frame_creation():
    # Create a frame with the default values
    origin = np.array([1, 2, 3]).reshape((3, 1))
    x_axis = np.array([1, 0, 0]).reshape((3, 1))
    y_axis = np.array([0, 1, 0]).reshape((3, 1))
    z_axis = np.array([0, 0, 1]).reshape((3, 1))
    frame = Frame(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)
    assert np.allclose(frame.origin, origin)
    assert np.allclose(frame.x_axis, x_axis)
    assert np.allclose(frame.y_axis, y_axis)
    assert np.allclose(frame.z_axis, z_axis)

def test_change_convention_frame():
    # Create a frame with the default values
    origin = np.array([1, 2, 3]).reshape((3, 1))
    x_axis = np.array([1, -1, 0]).reshape((3, 1))
    y_axis = np.array([1, 1, 0]).reshape((3, 1))
    z_axis = np.array([0, 0, 1]).reshape((3, 1))
    frame = Frame(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)

    # Change the convention
    R1 = frame.get_rotation(convention=1)
    T1 = frame.get_translation(convention=1)
    R7 = frame.get_rotation(convention=7)
    T7 = frame.get_translation(convention=7)

    # Compute the change
    R7_out, T7_out = switch_RT_convention(R1, T1, 1, 7)

    # Check the results
    assert np.allclose(R7.as_quat(), R7_out.as_quat())
    assert np.allclose(T7, T7_out)

def test_frame_parent():
    # Create a parent frame
    origin = np.array([1, 2, 3]).reshape((3, 1))
    x_axis = np.array([1, -1, 0]).reshape((3, 1))
    y_axis = np.array([1, 1, 0]).reshape((3, 1))
    z_axis = np.array([0, 0, 1]).reshape((3, 1))
    parent = Frame(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)

    # Create a child frame relative to the parent
    x_axis = np.array([1, 1, 0]).reshape((3, 1)) / np.sqrt(2)
    y_axis = np.array([-1, 1, 0]).reshape((3, 1)) / np.sqrt(2)
    z_axis = np.array([0, 0, 1]).reshape((3, 1))
    origin = - x_axis - 2 * y_axis - 3 * z_axis
    frame = Frame(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, parent=parent)

    # Get the global frame
    global_frame = frame.get_global_frame()

    # Check if the global frame is consistent
    assert np.allclose(global_frame.global_origin, np.array([0, 0, 0]).reshape((3, 1)))
    assert np.allclose(global_frame.global_axes, np.eye(3))

def test_set_and_get_rotation():
    # Create a frame with the default values
    origin = np.array([1, 2, 3]).reshape((3, 1))
    x_axis = np.array([1, -1, 0]).reshape((3, 1))
    y_axis = np.array([1, 1, 0]).reshape((3, 1))
    z_axis = np.array([0, 0, 1]).reshape((3, 1))
    frame = Frame(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)

    frame.set_rotation(Rotation.from_euler('xyz', [0, 0, np.pi / 2]), convention=0)
    R = frame.get_rotation(convention=0)

    assert np.allclose(R.as_euler('xyz'), [0, 0, np.pi / 2])

def test_set_and_get_translation_global():
    # Create a parent frame
    origin = np.array([1, 2, 3]).reshape((3, 1))
    x_axis = np.array([1, -1, 0]).reshape((3, 1))
    y_axis = np.array([1, 1, 0]).reshape((3, 1))
    z_axis = np.array([0, 0, 1]).reshape((3, 1))
    parent = Frame(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)

    # Create a child frame relative to the parent
    x_axis = np.array([1, 1, 0]).reshape((3, 1)) / np.sqrt(2)
    y_axis = np.array([-1, 1, 0]).reshape((3, 1)) / np.sqrt(2)
    z_axis = np.array([0, 0, 1]).reshape((3, 1))
    origin = - x_axis - 2 * y_axis - 3 * z_axis
    frame = Frame(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, parent=parent)

    frame.set_global_rotation(Rotation.from_euler('xyz', [np.pi / 3, 0, np.pi / 2]), convention=0)
    R = frame.get_global_rotation(convention=0)

    assert np.allclose(R.as_euler('xyz'), [np.pi / 3, 0, np.pi / 2])

def test_load_save():
    # Create a frame with the default values
    origin = np.array([1, 2, 3]).reshape((3, 1))
    x_axis = np.array([1, -1, 0]).reshape((3, 1)) / np.sqrt(2)
    y_axis = np.array([1, 1, 0]).reshape((3, 1)) / np.sqrt(2)
    z_axis = np.array([0, 0, 1]).reshape((3, 1))
    frame = Frame(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)

    # Save the frame
    data = frame.save_to_dict()

    # Load the frame
    frame_loaded = Frame.load_from_dict(data)

    # Check if the loaded frame is consistent
    assert np.allclose(frame_loaded.origin, origin)
    assert np.allclose(frame_loaded.x_axis, x_axis)
    assert np.allclose(frame_loaded.y_axis, y_axis)
    assert np.allclose(frame_loaded.z_axis, z_axis)
