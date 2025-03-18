import numpy as np
import pytest
from py3dframe import FrameTree, Frame

@pytest.fixture
def frame_tree():
    """Fixture to create a fresh FrameTree with some frames for each test."""
    tree = FrameTree()

    # Create frames with explicit origins and identity rotations.
    frame1 = Frame(origin=np.array([1, 0, 0]).reshape((3, 1)),
                   rotation_matrix=np.eye(3))
    frame2 = Frame(origin=np.array([0, 2, 0]).reshape((3, 1)),
                   rotation_matrix=np.eye(3))
    frame3 = Frame(origin=np.array([0, 0, 3]).reshape((3, 1)),
                   rotation_matrix=np.eye(3))

    # Add frames to the tree.
    tree.add_frame(frame1, "frame1")
    tree.add_frame(frame2, "frame2", parent="frame1")
    tree.add_frame(frame3, "frame3")

    return tree

def test_add_and_get_frame(frame_tree):
    expected_names = {"frame1", "frame2", "frame3"}
    assert set(frame_tree.names) == expected_names, "Frame names do not match expected names."

    assert frame_tree.get_parent_name("frame1") is None, "frame1 should have no parent."
    assert frame_tree.get_parent_name("frame2") == "frame1", "frame2 should have frame1 as parent."
    assert frame_tree.get_parent_name("frame3") is None, "frame3 should have no parent."

    f1 = frame_tree.get_frame("frame1")
    assert f1 == frame_tree["frame1"], "get_frame did not return the correct frame."

def test_set_name(frame_tree):
    frame_tree.set_name("frame1", "base")
    assert "base" in frame_tree.names, "New name 'base' not found in tree names."
    assert "frame1" not in frame_tree.names, "Old name 'frame1' should be removed."
    assert frame_tree.get_parent_name("base") is None, "base should have no parent."

def test_set_parent(frame_tree):
    assert frame_tree.get_parent_name("frame3") is None, "frame3 should have no parent."
    frame_tree.set_parent("frame3", "frame1")
    assert frame_tree.get_parent_name("frame3") == "frame1", "frame3 should have frame1 as parent."

def test_remove_frame(frame_tree):
    frame_tree.remove_frame("frame1")
    assert "frame1" not in frame_tree.names, "frame1 was not removed."
    assert frame_tree.get_parent_name("frame2") is None, "frame2 should have no parent after frame1 removal."

def test_recursive_remove_frame(frame_tree):
    frame4 = Frame(origin=np.array([0, 0, 1]).reshape((3, 1)), rotation_matrix=np.eye(3))
    frame_tree.add_frame(frame4, "frame4", parent="frame2")

    frame_tree.recursive_remove_frame("frame2")
    assert "frame2" not in frame_tree.names, "frame2 was not removed recursively."
    assert "frame4" not in frame_tree.names, "Child frame4 was not removed recursively."

def test_get_globalcompose_frame(frame_tree):
    global_point = np.array([1, 2, -1], dtype=np.float64)
    point_frame_1 = frame_tree["frame1"].from_parent_to_frame(point=global_point)
    point_frame_2 = frame_tree["frame2"].from_parent_to_frame(point=point_frame_1)
    composed = frame_tree.get_global_frame("frame2")
    point_composed = composed.from_global_to_frame(point=global_point)
    assert np.allclose(point_frame_2, point_composed, atol=1e-6), \
        "Composed frame does not match the expected transformation."

def test_get_composed_frame(frame_tree):
    composed = frame_tree.get_composed_frame(input_name="frame1", output_name="frame2")
    assert composed == frame_tree["frame2"], "Composed frame does not match frame2."

def test_from_frame_to_frame(frame_tree):
    point_in_frame1 = np.array([[10], [10], [10]], dtype=np.float64)
    converted = frame_tree.from_frame_to_frame(input_name="frame1", output_name="frame2", point=point_in_frame1)
    expected_point = point_in_frame1 - np.array([[0], [2], [0]], dtype=np.float64)
    assert np.allclose(converted, expected_point, atol=1e-6), \
        "Conversion from frame1 to frame2 did not yield the expected point."

def test_clear(frame_tree):
    frame_tree.clear()
    assert len(frame_tree.names) == 0, "FrameTree should be empty after clear()."
    assert len(frame_tree._name_to_uuid) == 0, "Name to UUID map should be empty after clear()."
    assert len(frame_tree._uuid_to_name) == 0, "UUID to name map should be empty after clear()."


def test_save_load_json(tmp_path, frame_tree):
    """Test saving and loading a FrameTree to/from a JSON file."""
    filepath = tmp_path / "test_frametree.json"
    
    frame_tree.save_to_json(filepath)
    loaded_tree = FrameTree.load_from_json(filepath)

    assert set(loaded_tree.names) == set(frame_tree.names), \
        "Loaded JSON tree does not have the correct frame names."
    for name in frame_tree.names:
        if frame_tree.get_parent_name(name) is not None:
            assert loaded_tree.get_parent_name(name) == frame_tree.get_parent_name(name) 
        else:
            assert loaded_tree.get_parent_name(name) is None

    for name in frame_tree.names:
        assert np.allclose(frame_tree[name].origin, loaded_tree[name].origin, atol=1e-6), \
            f"Loaded JSON frame '{name}' origin does not match."
        assert np.allclose(frame_tree[name].rotation_matrix, loaded_tree[name].rotation_matrix, atol=1e-6), \
            f"Loaded JSON frame '{name}' rotation matrix does not match."
