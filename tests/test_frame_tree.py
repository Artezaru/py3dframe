import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from py3dframe import Frame, FrameTree, FrameTransform

def test_instantiation():
    """ Test that FrameTree can be instantiated without errors. """
    ft = FrameTree()
    assert isinstance(ft, FrameTree), "FrameTree instantiation failed."
    
def test_instantiate_with_root_frame():
    """ Test that FrameTree can be instantiated with a root Frame. """
    root_frame = Frame.from_quaternion(
        translation=np.array([[3], [2], [0]]),
        quaternion=np.array([0.1, 0.2, 0.3, 0.4])
    )
    ft = FrameTree(root_frame=root_frame)
    assert isinstance(ft, FrameTree), "FrameTree instantiation with root frame failed."
    assert ft['root'] is root_frame, "Root frame not set correctly in FrameTree."
    
def test_connect_frames():
    """ Test connecting two frames in the FrameTree. """
    ft = FrameTree()
    
    frame_a = Frame.from_euler_angles(
        translation=np.array([[1], [0], [0]]),
        euler_angles=np.array([45, 0, 0]),
        degrees=True
    )
    
    frame_b = Frame.from_euler_angles(
        translation=np.array([[0], [1], [0]]),
        euler_angles=np.array([0, 45, 0]),
        degrees=True
    )
    
    ft.connect_frame('frame_a', frame_a, parent_name='root')
    ft.connect_frame('frame_b', frame_b, parent_name='frame_a')
    
    assert ft.get_frame('frame_a') is frame_a, "Frame A not connected properly."
    assert ft.get_frame('frame_b') is frame_b, "Frame B not connected properly."
    assert ft['frame_b'].parent is ft['frame_a'], "Frame B's parent is incorrect."
    
def test_disconnect_frames():
    """ Test disconnecting a frame from the FrameTree. """
    ft = FrameTree()
    
    frame_a = Frame.from_euler_angles(
        translation=np.array([[1], [0], [0]]),
        euler_angles=np.array([45, 0, 0]),
        degrees=True
    )
    
    ft.connect_frame('frame_a', frame_a, parent_name='root')
    ft.disconnect_frame('frame_a')
    
    assert 'frame_a' not in ft, "Frame A was not disconnected properly."
    
def test_get_child_names():
    """ Test retrieving the names of child frames. """
    ft = FrameTree()
    
    frame_a = Frame.from_euler_angles(
        translation=np.array([[1], [0], [0]]),
        euler_angles=np.array([45, 0, 0]),
        degrees=True
    )
    
    frame_b = Frame.from_euler_angles(
        translation=np.array([[0], [1], [0]]),
        euler_angles=np.array([0, 45, 0]),
        degrees=True
    )
    
    frame_c = Frame.from_euler_angles(
        translation=np.array([[0], [0], [1]]),
        euler_angles=np.array([0, 0, 45]),
        degrees=True
    )
    
    ft.connect_frame('frame_a', frame_a, parent_name='root')
    ft.connect_frame('frame_b', frame_b, parent_name='frame_a')
    ft.connect_frame('frame_c', frame_c, parent_name='frame_a')
    
    childs_of_root = ft.get_child_names('root')
    childs_of_a = ft.get_child_names('frame_a')
    
    assert childs_of_root == ['frame_a'], "Child frames of root are incorrect."
    assert childs_of_a == ['frame_b', 'frame_c'], "Child frames of frame A are incorrect."
    
def test_list_frames():
    """ Test listing all frames in the FrameTree. """
    ft = FrameTree()
    
    frame_a = Frame.from_euler_angles(
        translation=np.array([[1], [0], [0]]),
        euler_angles=np.array([45, 0, 0]),
        degrees=True
    )
    
    frame_b = Frame.from_euler_angles(
        translation=np.array([[0], [1], [0]]),
        euler_angles=np.array([0, 45, 0]),
        degrees=True
    )
    
    ft.connect_frame('frame_a', frame_a, parent_name='root')
    ft.connect_frame('frame_b', frame_b, parent_name='frame_a')
    
    frame_names = ft.list_frames()
    
    assert set(frame_names) == {'root', 'frame_a', 'frame_b'}, "Frame listing is incorrect."
    
def test_get_parent_name():
    """ Test retrieving the parent name of a frame. """
    ft = FrameTree()
    
    frame_a = Frame.from_euler_angles(
        translation=np.array([[1], [0], [0]]),
        euler_angles=np.array([45, 0, 0]),
        degrees=True
    )
    
    ft.connect_frame('frame_a', frame_a, parent_name='root')
    
    parent_name = ft.get_parent_name('frame_a')
    
    assert parent_name == 'root', "Parent name retrieval is incorrect."
    
def test_rename_frame():
    """ Test renaming a frame in the FrameTree. """
    ft = FrameTree()
    
    frame_a = Frame.from_euler_angles(
        translation=np.array([[1], [0], [0]]),
        euler_angles=np.array([45, 0, 0]),
        degrees=True
    )
    
    ft.connect_frame('frame_a', frame_a, parent_name='root')
    ft.rename_frame('frame_a', 'renamed_frame_a')
    
    assert 'renamed_frame_a' in ft, "Frame renaming failed."
    assert 'frame_a' not in ft, "Old frame name still exists after renaming."
    assert ft.get_frame('renamed_frame_a') is frame_a, "Renamed frame retrieval failed."
    
def test_disconnect_frame_recursive():
    """ Test disconnecting a frame with children recursively. """
    ft = FrameTree()
    
    frame_a = Frame.from_euler_angles(
        translation=np.array([[1], [0], [0]]),
        euler_angles=np.array([45, 0, 0]),
        degrees=True
    )
    
    frame_b = Frame.from_euler_angles(
        translation=np.array([[0], [1], [0]]),
        euler_angles=np.array([0, 45, 0]),
        degrees=True
    )
    
    ft.connect_frame('frame_a', frame_a, parent_name='root')
    ft.connect_frame('frame_b', frame_b, parent_name='frame_a')
    
    ft.disconnect_frame('frame_a', recursive=True)
    
    assert 'frame_a' not in ft, "Frame A was not disconnected properly."
    assert 'frame_b' not in ft, "Child Frame B was not disconnected properly."
    
def test_replace_frame():
    """ Test replacing a frame in the FrameTree. """
    ft = FrameTree()
    
    frame_a = Frame.from_euler_angles(
        translation=np.array([[1], [0], [0]]),
        euler_angles=np.array([45, 0, 0]),
        degrees=True
    )
    
    frame_b = Frame.from_euler_angles(
        translation=np.array([[0], [1], [0]]),
        euler_angles=np.array([0, 45, 0]),
        degrees=True
    )
    
    new_frame_a = Frame.from_euler_angles(
        translation=np.array([[2], [0], [0]]),
        euler_angles=np.array([90, 0, 0]),
        degrees=True
    )
    
    ft.connect_frame('frame_a', frame_a, parent_name='root')
    ft.connect_frame('frame_b', frame_b, parent_name='frame_a')
    ft.replace_frame('frame_a', new_frame_a)
    
    assert ft.get_frame('frame_a') is new_frame_a, "Frame replacement failed."
    assert ft.get_frame('frame_a').translation[0, 0] == 2, "New frame translation is incorrect."
    assert ft.get_frame('frame_b').parent is new_frame_a, "Child frame's parent not updated after replacement."
    
def test_set_root_frame():
    """ Test setting a new root frame in the FrameTree. """
    ft = FrameTree()
    
    new_root_frame = Frame.from_euler_angles(
        translation=np.array([[0], [0], [0]]),
        euler_angles=np.array([0, 0, 0]),
        degrees=True
    )
    
    ft.set_root_frame(new_root_frame)
    
    assert ft['root'] is new_root_frame, "Setting new root frame failed."
    
def test_move_frame():
    """ Test moving a frame to a new parent in the FrameTree. """
    ft = FrameTree()
    
    frame_a = Frame.from_euler_angles(
        translation=np.array([[1], [0], [0]]),
        euler_angles=np.array([45, 0, 0]),
        degrees=True
    )
    
    frame_b = Frame.from_euler_angles(
        translation=np.array([[0], [1], [0]]),
        euler_angles=np.array([0, 45, 0]),
        degrees=True
    )
    
    ft.connect_frame('frame_a', frame_a, parent_name='root')
    ft.connect_frame('frame_b', frame_b, parent_name='root')
    
    ft.move_frame('frame_b', new_parent_name='frame_a')
    
    assert ft['frame_b'].parent is ft['frame_a'], "Moving frame to new parent failed."
    
def test_get_transform():
    """ Test retrieving the transform between two frames in the FrameTree. """
    ft = FrameTree()
    
    frame_a = Frame.from_euler_angles(
        translation=np.array([[1], [0], [0]]),
        euler_angles=np.array([45, 0, 0]),
        degrees=True
    )
    
    frame_b = Frame.from_euler_angles(
        translation=np.array([[0], [1], [0]]),
        euler_angles=np.array([0, 45, 0]),
        degrees=True
    )
    
    ft.connect_frame('frame_a', frame_a, parent_name='root')
    ft.connect_frame('frame_b', frame_b, parent_name='frame_a')
    
    transform = ft.get_transform('root', 'frame_b')
    
    expected_transform = FrameTransform(
        input_frame=ft['root'],
        output_frame=ft['frame_b']
    )
    assert expected_transform.input_frame is transform.input_frame, "Input frame of transform is incorrect."
    assert expected_transform.output_frame is transform.output_frame, "Output frame of transform is incorrect."
    
def test_print_tree():
    """ Test printing the FrameTree structure. """
    ft = FrameTree()
    
    frame_a = Frame.from_euler_angles(
        translation=np.array([[1], [0], [0]]),
        euler_angles=np.array([45, 0, 0]),
        degrees=True
    )
    
    frame_b = Frame.from_euler_angles(
        translation=np.array([[0], [1], [0]]),
        euler_angles=np.array([0, 45, 0]),
        degrees=True
    )
    
    ft.connect_frame('frame_a', frame_a, parent_name='root')
    ft.connect_frame('frame_b', frame_b, parent_name='frame_a')
    
    ft.print_tree()  # Just ensure this runs without error