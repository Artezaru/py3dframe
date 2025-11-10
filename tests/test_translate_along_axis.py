import pytest
import numpy as np
from py3dframe import Frame
from py3dframe.manipulations import translate_along_axis


def create_test_frame():
    """Helper function to create a more complex test frame."""
    frame = Frame.from_axes(
        origin=[1.0, 2.0, 3.0],
        x_axis=[0.0, 1.0, 1.0],
        y_axis=[0.0, 1.0, -1.0],
        z_axis=[-1.0, 0.0, 0.0],
    )
    parent_frame = Frame.from_axes(
        origin=[10.0, 10.0, 10.0],
        x_axis=[0.0, 0.0, 1.0],
        y_axis=[1.0, 0.0, 0.0],
        z_axis=[0.0, 1.0, 0.0],
    )
    frame.parent = parent_frame
    return frame


def test_translate_local():
    frame = create_test_frame()
    translated_frame = translate_along_axis(frame, axis=[1.0, 1.0, 0.0], distance=1.0, use_global=False, inplace=False)

    expected_origin_local = np.array([1.70710678, 2.70710678, 3.0]).reshape((3, 1))
    assert np.allclose(translated_frame.origin, expected_origin_local)


def test_translate_global():
    frame = create_test_frame()
    translated_frame = translate_along_axis(frame, axis=[1.0, 1.0, 0.0], distance=1.0, use_global=True, inplace=False)

    # dx global -> dy for parent
    # dy global -> dz for parent
    # dz global -> dx for parent

    expected_delta_origin_local = np.array([0.0, 0.70710678, 0.70710678]).reshape((3, 1))
    
    expected_origin_local = frame.origin + expected_delta_origin_local
    assert np.allclose(translated_frame.origin, expected_origin_local)