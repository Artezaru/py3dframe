
import pytest
import numpy as np
from py3dframe import Frame
from py3dframe.manipulations import mirror_across_plane



def test_double_mirror():
    origin = np.array([1.0, 2.0, 3.0])
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    frame = Frame.from_axes(origin, x_axis, y_axis, z_axis)

    # Define the normal of the plane (e.g., the xy-plane)
    normal = np.array([0.0, 0.0, 1.0])
    point = np.array([0.0, 0.0, 0.0])  # A point on the plane

    # Mirror the frame across the specified plane in the local coordinate system
    mirrored_frame = mirror_across_plane(frame, normal, point, use_global=False, inplace=False)

    # Mirror again to return to original
    double_mirrored_frame = mirror_across_plane(mirrored_frame, normal, point, use_global=False, inplace=False)

    assert np.allclose(double_mirrored_frame.origin, frame.origin)
    assert np.allclose(double_mirrored_frame.x_axis, frame.x_axis)
    assert np.allclose(double_mirrored_frame.y_axis, frame.y_axis)
    assert np.allclose(double_mirrored_frame.z_axis, frame.z_axis)