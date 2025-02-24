import unittest
import numpy as np
from py3dframe import FrameTree, Frame

class TestFrameTree(unittest.TestCase):

    def setUp(self):
        # Create a fresh FrameTree for each test.
        self.tree = FrameTree()

        # Create three frames with explicit origins and identity rotations.
        # Frame1: translation [1, 0, 0]
        self.frame1 = Frame(origin=np.array([1, 0, 0]).reshape((3, 1)),
                            rotation_matrix=np.eye(3))
        # Frame2: translation [0, 2, 0] (relative to its parent)
        self.frame2 = Frame(origin=np.array([0, 2, 0]).reshape((3, 1)),
                            rotation_matrix=np.eye(3))
        # Frame3: translation [0, 0, 3]
        self.frame3 = Frame(origin=np.array([0, 0, 3]).reshape((3, 1)),
                            rotation_matrix=np.eye(3))

        # Add frames to the tree.
        self.tree.add_frame(self.frame1, "frame1")
        # Make frame2 a child of frame1.
        self.tree.add_frame(self.frame2, "frame2", parent="frame1")
        # Frame3 remains independent (i.e., parent is None, meaning world).
        self.tree.add_frame(self.frame3, "frame3")

    def test_add_and_get_frame(self):
        # Explicitly reset the tree.
        self.setUp()
        # Verify names and frames dictionaries.
        expected_names = {"frame1", "frame2", "frame3"}
        self.assertEqual(set(self.tree.names), expected_names,
                         "Frame names do not match expected names.")
        # Check parents: frame1 and frame3 should have no parent (None),
        # while frame2's parent should be 'frame1'.
        self.assertIsNone(self.tree.parents["frame1"],
                          "frame1 should have no parent.")
        self.assertEqual(self.tree.parents["frame2"], "frame1",
                         "frame2 should have frame1 as parent.")
        self.assertIsNone(self.tree.parents["frame3"],
                          "frame3 should have no parent.")
        # Test get_frame (without copy).
        f1 = self.tree.get_frame("frame1")
        self.assertEqual(f1, self.frame1, "get_frame did not return the correct frame.")

    def test_set_name(self):
        # Explicitly reset the tree.
        self.setUp()
        # Rename frame1 to 'base'.
        self.tree.set_name("frame1", "base")
        self.assertIn("base", self.tree.names, "New name 'base' not found in tree names.")
        self.assertNotIn("frame1", self.tree.names, "Old name 'frame1' should be removed.")
        # Frame2's parent should now be updated to 'base'.
        self.assertEqual(self.tree.parents["frame2"], "base",
                         "Parent reference was not updated correctly after renaming.")

    def test_set_parent(self):
        # Explicitly reset the tree.
        self.setUp()
        # Initially, frame3 should have no parent.
        self.assertIsNone(self.tree.parents["frame3"], "frame3 should initially have no parent.")
        # Set frame3's parent to frame1.
        self.tree.set_parent("frame3", "frame1")
        self.assertEqual(self.tree.parents["frame3"], "frame1",
                         "Setting frame3's parent to frame1 failed.")

    def test_remove_frame(self):
        # Explicitly reset the tree.
        self.setUp()
        # Remove frame1. This should update frame2 (whose parent was frame1) to have no parent.
        self.tree.remove_frame("frame1")
        self.assertNotIn("frame1", self.tree.names, "frame1 was not removed.")
        self.assertIsNone(self.tree.parents["frame2"],
                          "frame2's parent should be set to None after removing its parent.")

    def test_recursive_remove_frame(self):
        # Explicitly reset the tree.
        self.setUp()
        # Create a deeper hierarchy: let frame4 be a child of frame2.
        frame4 = Frame(origin=np.array([0, 0, 1]).reshape((3, 1)),
                       rotation_matrix=np.eye(3))
        self.tree.add_frame(frame4, "frame4", parent="frame2")
        # Now, recursively remove frame2: This should remove both frame2 and frame4.
        self.tree.recursive_remove_frame("frame2")
        self.assertNotIn("frame2", self.tree.names, "frame2 was not removed recursively.")
        self.assertNotIn("frame4", self.tree.names, "Child frame4 was not removed recursively.")

    def test_get_worldcompose_frame(self):
        # Explicitly reset the tree.
        self.setUp()
        # For frame2, which is a child of frame1:
        # frame1 has origin [1, 0, 0] and identity rotation.
        # frame2 has origin [0, 2, 0] relative to its parent.
        # Composing these should yield a frame with origin [1, 2, 0] and identity rotation.
        world_point = np.array([1, 2, -1], dtype=np.float32)
        point_local_1 = self.tree["frame1"].from_world_to_local(point=world_point)
        point_local_2 = self.tree["frame2"].from_world_to_local(point=point_local_1)
        composed = self.tree.get_worldcompose_frame("frame2")
        point_composed = composed.from_world_to_local(point=world_point)
        self.assertTrue(np.allclose(point_local_2, point_composed, atol=1e-6),
                        "Composed frame does not match the expected transformation.")

    def test_get_compose_frame(self):
        # Explicitly reset the tree.
        self.setUp()
        # The composed from frame1 to frame2 should return the same as frame2.
        composed = self.tree.get_compose_frame(input_name="frame1", output_name="frame2")
        self.assertEqual(composed, self.tree["frame2"], "Composed frame does not match frame2.")

    def test_from_frame_to_frame(self):
        # Explicitly reset the tree.
        self.setUp()
        # Test conversion of a point from frame1 to frame2.
        point_in_frame1 = np.array([[10], [10], [10]], dtype=np.float32)
        converted = self.tree.from_frame_to_frame(input_name="frame1", output_name="frame2", point=point_in_frame1)
        # In our simple example, since both frames have identity rotation,
        # the relative translation from frame1 ([1, 0, 0] world origin) to frame2 ([1, 2, 0] worldcompose origin)
        # is [0, 2, 0]. Thus, the conversion should subtract [0, 2, 0] from the input.
        expected_point = point_in_frame1 - np.array([[0], [2], [0]], dtype=np.float32)
        self.assertTrue(np.allclose(converted, expected_point, atol=1e-6),
                        "Conversion from frame1 to frame2 did not yield the expected point.")

    def test_clear(self):
        # Explicitly reset the tree.
        self.setUp()
        self.tree.clear()
        self.assertEqual(len(self.tree.names), 0, "FrameTree should be empty after clear().")
        self.assertEqual(len(self.tree.parents), 0, "Parent links should be empty after clear().")

if __name__ == '__main__':
    unittest.main()
