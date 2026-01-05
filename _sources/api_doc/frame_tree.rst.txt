.. currentmodule:: py3dframe

py3dframe.FrameTree
=========================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

.. autoclass:: FrameTree

Update the tree connections
--------------------------------

The connections between frames in the tree can be set using the :meth:`connect_frame` method.

.. autosummary::
   :toctree: ../generated/

    FrameTree.connect_frame
    FrameTree.disconnect_frame
    FrameTree.list_frames
    FrameTree.get_child_names
    FrameTree.get_frame
    FrameTree.get_parent_name
    FrameTree.move_frame
    FrameTree.rename_frame
    FrameTree.replace_frame
    FrameTree.set_root_frame


Transformations between frames in the tree
--------------------------------------------

.. autosummary::
   :toctree: ../generated/

    FrameTree.get_transform
    FrameTree.transform

Save and load FrameTree objects
--------------------------------

.. autosummary::
   :toctree: ../generated/

    FrameTree.to_dict
    FrameTree.from_dict
    FrameTree.to_json
    FrameTree.from_json


Additional methods
--------------------------------

.. autosummary::
   :toctree: ../generated/

    FrameTree.__repr__
    FrameTree.__str__
    FrameTree.__len__
    FrameTree.__contains__
    FrameTree.__getitem__
    FrameTree.__bool__
    FrameTree.print_tree

Usage example
----------------------------

Create some frames and build a FrameTree

.. code-block:: python

    import numpy as np
    from py3dframe import Frame, FrameTree, Rotation

    # Create some frames
    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    translation = np.array([0, 0, 0]).reshape(3, 1)

    root_frame = Frame.from_rotation(translation=translation, rotation=rotation, convention=0)

    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    translation = np.array([1, 0, 0]).reshape(3, 1)

    child_frame_1 = Frame.from_rotation(translation=translation, rotation=rotation, convention=0)

    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    translation = np.array([0, 1, 0]).reshape(3, 1)

    child_frame_2 = Frame.from_rotation(translation=translation, rotation=rotation, convention=0)

    # Build a FrameTree
    frame_tree = FrameTree()
    frame_tree.connect_frame(name="Root_Frame", frame=root_frame)
    frame_tree.connect_frame(name="Child_Frame_1", frame=child_frame_1, parent_name="Root_Frame")
    frame_tree.connect_frame(name="Child_Frame_2", frame=child_frame_2, parent_name="Root_Frame")

    # Print the FrameTree
    frame_tree.print_tree()

