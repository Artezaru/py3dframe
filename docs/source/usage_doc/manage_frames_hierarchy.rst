Managing frames hierarchy
=========================

Basic usage
-----------

Lets consider three frames :math:`F_1`, :math:`F_2` and :math:`F_3` with the following hierarchy:

The frame :math:`F_1` is the relation between the global frame and a space :math:`S_1` of :math:`\mathbb{R}^3`. 
The frame :math:`F_2` is the relation between the space :math:`S_1` and a space :math:`S_2` of :math:`\mathbb{R}^3`. 
The frame :math:`F_3` is the relation between the global frame and a space :math:`S_3` of :math:`\mathbb{R}^3`.

.. code-block:: console

    global
    ├── frame1
    │   └── frame2
    └── frame3

To easily convert points from :math:`S_3` to :math:`S_2`, you can implement the following frame hierarchy:

.. code-block:: python

    import py3dframe

    frame1 = py3dframe.Frame()
    frame2 = py3dframe.Frame()
    frame3 = py3dframe.Frame()

    frame_tree = py3dframe.FrameTree()

    frame_tree.add_frame(frame1, name='F1', parent=None)
    frame_tree.add_frame(frame2, name='F2', parent='F1')
    frame_tree.add_frame(frame3, name='F3', parent=None)

The attribute `parent` of the method `add_frame` is used to link the frame to another frame by its name.
By default, the link is set to `None`, which means that the frame is linked to the global frame.

To convert a point from :math:`S_3` to :math:`S_2`, you can use the following code:

.. code-block:: python

    import py3dframe

    frame1 = py3dframe.Frame()
    frame2 = py3dframe.Frame()
    frame3 = py3dframe.Frame()

    frame_tree = py3dframe.FrameTree()

    frame_tree.add_frame(frame1, name='F1', link=None)
    frame_tree.add_frame(frame2, name='F2', link='F1')
    frame_tree.add_frame(frame3, name='F3', link=None)

    point_S3 = [1, 1, 1]
    point_S2 = frame_tree.from_frame_to_frame(input_name='F3', output_name='F2', point=point_S3)

    print(point_S2)






