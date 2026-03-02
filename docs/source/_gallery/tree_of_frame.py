"""
.. _example_tree_of_frames:

Deals with numerous frames (Tree of frames)
===============================================================

In this example, we will see how to define a tree of frames, where each frame is defined with respect to another frame.

When managing a large number of frames, it is often useful to organize them in a tree structure better than defining the architecture manually.

"""

# %%
# Create a tree of frames
# ---------------------------------------------------
#
# By default the ``"root"`` frame is the canonical frame, but it can be defined as any frame using :meth:`py3dframe.FrameTree.set_root_frame`.
#
# Then the frames can be connected to the tree using :meth:`py3dframe.FrameTree.connect_frame`, where the name of the parent frame must be specified and ``"root"`` refers to the root frame.

import numpy as np
from py3dframe import Frame, FrameTree, Rotation

# Create some frames
rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
translation = np.array([0, 0, 0]).reshape(3, 1)

root_frame = Frame.from_rotation(translation=translation, rotation=rotation, convention=0)

rotation = Rotation.from_euler('xyz', [0, 15, 45], degrees=True)
translation = np.array([1, 0, 0]).reshape(3, 1)

child_frame_1 = Frame.from_rotation(translation=translation, rotation=rotation, convention=0)

rotation = Rotation.from_euler('xyz', [12, 0, 0], degrees=True)
translation = np.array([0, 1, 0]).reshape(3, 1)

child_frame_2 = Frame.from_rotation(translation=translation, rotation=rotation, convention=0)

rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
translation = np.array([0, 0, 1]).reshape(3, 1)

child_frame_3 = Frame.from_rotation(translation=translation, rotation=rotation, convention=0)

# Build a FrameTree
frame_tree = FrameTree()
frame_tree.set_root_frame(root_frame)
frame_tree.connect_frame(name="Child_Frame_1", frame=child_frame_1, parent_name="root")
frame_tree.connect_frame(name="Child_Frame_2", frame=child_frame_2, parent_name="root")
frame_tree.connect_frame(name="Child_Frame_3", frame=child_frame_3, parent_name="Child_Frame_1")

# Print the FrameTree
frame_tree.print_tree()


# %%
# Perform transformations between frames in the tree
# ---------------------------------------------------
#
# The transformations between the frames in the tree can be performed using 
# the method ``transform`` of the class :class:`py3dframe.FrameTree`.
#
# The points and vectors must be :math:`(3, N)` numpy arrays, where :math:`N` is the number of points or vectors to transform.
#

point_child1 = np.array([1, 2, 3]).reshape(3, 1)
point_child2 = frame_tree.transform(
    "Child_Frame_1", 
    "Child_Frame_2", 
    point=point_child1
)
print(f"Point in Child_Frame_1: {point_child1.flatten()}")
print(f"Point in Child_Frame_2: {point_child2.flatten()}")

vector_child1 = np.array([1, 0, 0]).reshape(3, 1)
vector_child2 = frame_tree.transform(
    "Child_Frame_1", 
    "Child_Frame_2", 
    vector=vector_child1
)
print(f"Vector in Child_Frame_1: {vector_child1.flatten()}")
print(f"Vector in Child_Frame_2: {vector_child2.flatten()}")

# %%
# Save and load the FrameTree (JSON format)
# ---------------------------------------------------
#
# The FrameTree can be saved and loaded in JSON format using the methods 
# ``to_json`` and ``from_json`` of the class :class:`py3dframe.FrameTree`.

json_file = "frame_tree.json"
frame_tree.to_json(json_file)
loaded_frame_tree = FrameTree.from_json(json_file)
loaded_frame_tree.print_tree()

