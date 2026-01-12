import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from py3dframe import Frame, FrameTree

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