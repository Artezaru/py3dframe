
import pytest
import numpy as np
from py3dframe import Frame
from py3dframe.manipulations import rotate_around_axis



def test_config_1():
    origin = np.array([3.0, 0.0, 0.0])
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    frame = Frame.from_axes(origin, x_axis, y_axis, z_axis)

    # Rotation of 90 and 125 degrees around z-axis at point (0,0,0)
    #
    # -> z-axis must remain unchanged
    # -> x-axis and y-axis must rotate accordingly by 90 and 125 degrees
    # -> origin must rotate accordingly by 90 and 125 degrees

    rotated_frame = rotate_around_axis(
        frame,
        axis=[0.0, 0.0, 1.0],
        angle=90*np.pi/180,
        point=[0.0, 0.0, 0.0],
        use_global=True,
        inplace=False,
    )

    expected_origin = np.array([0.0, 3.0, 0.0]).reshape((3, 1))
    expected_x_axis = np.array([0.0, 1.0, 0.0]).reshape((3, 1))
    expected_y_axis = np.array([-1.0, 0.0, 0.0]).reshape((3, 1))
    expected_z_axis = np.array([0.0, 0.0, 1.0]).reshape((3, 1))

    assert np.allclose(rotated_frame.origin, expected_origin)
    assert np.allclose(rotated_frame.x_axis, expected_x_axis)
    assert np.allclose(rotated_frame.y_axis, expected_y_axis)
    assert np.allclose(rotated_frame.z_axis, expected_z_axis)

    rotated_frame = rotate_around_axis(
        frame,
        axis=[0.0, 0.0, 1.0],
        angle=125*np.pi/180,
        point=[0.0, 0.0, 0.0],
        use_global=True,
        inplace=False,
    )

    expected_origin = np.array([-0.57357644*3, 0.81915204*3, 0.0]).reshape((3, 1))
    expected_x_axis = np.array([-0.57357644, 0.81915204, 0.0]).reshape((3, 1))
    expected_y_axis = np.array([-0.81915204, -0.57357644, 0.0]).reshape((3, 1))
    expected_z_axis = np.array([0.0, 0.0, 1.0]).reshape((3, 1))

    assert np.allclose(rotated_frame.origin, expected_origin)
    assert np.allclose(rotated_frame.x_axis, expected_x_axis)
    assert np.allclose(rotated_frame.y_axis, expected_y_axis)
    assert np.allclose(rotated_frame.z_axis, expected_z_axis)