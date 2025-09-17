import pytest
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
from py3dframe import Frame, FrameTransform

def test_transform_initialization():
    frame_E = Frame.canonical()
    frame_F = Frame.from_axes(origin=[1, 2, 3], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1])
    transform = FrameTransform(input_frame=frame_E, output_frame=frame_F, dynamic=True, convention=0)

    assert transform.input_frame == frame_E
    assert transform.output_frame == frame_F
    assert transform.dynamic is True
    assert transform.convention == 0

def test_rotation_matrix():
    frame_E = Frame.canonical()
    frame_F = Frame.from_axes(origin=[1, 2, 3], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1])
    transform = FrameTransform(input_frame=frame_E, output_frame=frame_F, dynamic=True, convention=0)

    rotation_matrix_expected = np.eye(3)
    rotation_matrix_actual = transform.rotation_matrix

    assert np.allclose(rotation_matrix_actual, rotation_matrix_expected)

def test_translation_vector():
    frame_E = Frame.canonical()
    frame_F = Frame.from_axes(origin=[1, 2, 3], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1])
    transform = FrameTransform(input_frame=frame_E, output_frame=frame_F, dynamic=True, convention=0)

    translation_expected = np.array([[1], [2], [3]])
    translation_actual = transform.translation

    assert np.allclose(translation_actual, translation_expected)

def test_transform_point():
    frame_E = Frame.canonical()
    frame_F = Frame.from_axes(origin=[1, 2, 3], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1])
    transform = FrameTransform(input_frame=frame_E, output_frame=frame_F, dynamic=True, convention=0)

    point_E = np.array([4, 5, 6]).reshape((3, 1))
    point_F = transform.transform(point=point_E)

    expected_point_F = np.array([[3], [3], [3]])
    assert np.allclose(point_F, expected_point_F)

def test_inverse_transform_point():
    frame_E = Frame.canonical()
    frame_F = Frame.from_axes(origin=[1, 2, 3], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1])
    transform = FrameTransform(input_frame=frame_E, output_frame=frame_F, dynamic=True, convention=0)

    point_F = np.array([1, 2, 3]).reshape((3, 1))
    point_E = transform.inverse_transform(point=point_F)

    expected_point_E = np.array([[2], [4], [6]])
    assert np.allclose(point_E, expected_point_E)

def test_transform_vector():
    frame_E = Frame.canonical()
    frame_F = Frame.from_axes(origin=[1, 2, 3], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1])
    transform = FrameTransform(input_frame=frame_E, output_frame=frame_F, dynamic=True, convention=0)

    vector_E = np.array([1, 0, 0]).reshape((3, 1))
    vector_F = transform.transform(vector=vector_E)

    expected_vector_F = np.array([[1], [0], [0]])
    assert np.allclose(vector_F, expected_vector_F)

def test_invalid_convention():
    frame_E = Frame.canonical()
    frame_F = Frame.from_axes(origin=[1, 2, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1])
    with pytest.raises(ValueError):
        FrameTransform(input_frame=frame_E, output_frame=frame_F, dynamic=True, convention=8)

def test_transform_point_dynamique():
    frame_E = Frame.canonical()
    frame_F = Frame.from_axes(origin=[1, 2, 0], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1])
    frame_G = Frame.from_axes(origin=[0, 0, 0], x_axis=[1, -1, 0], y_axis=[1, 1, 0], z_axis=[0, 0, 1], parent=frame_F)
    transform = FrameTransform(input_frame=frame_E, output_frame=frame_G, dynamic=True, convention=0)

    point_E = np.array([4, 5, 6]).reshape((3, 1))
    point_G = transform.transform(point=point_E)

    expected_point_G = np.array([[0], [3 * np.sqrt(2)], [6]])
    assert np.allclose(point_G, expected_point_G)

    # Lets move F
    frame_F.origin = np.array([1, 4, 0])

    point_G = transform.transform(point=point_E)
    expected_point_G = np.array([[np.sqrt(2)], [2 * np.sqrt(2)], [6]])

    # Lets again move F without dynamic
    transform.dynamic = False
    frame_F.origin = np.array([1, 2, 0])

    point_G = transform.transform(point=point_E)
    expected_point_G = np.array([[np.sqrt(2)], [2 * np.sqrt(2)], [6]])
    
    assert np.allclose(point_G, expected_point_G)

    # Lets move F again with dynamic
    transform.dynamic = True

    point_G = transform.transform(point=point_E)
    expected_point_G = np.array([[0], [3 * np.sqrt(2)], [6]])

    assert np.allclose(point_G, expected_point_G)
    
