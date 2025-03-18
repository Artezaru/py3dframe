How to use a Frame
======================

Basic usage
-----------

Lets consider a simple frame of :math:`\mathbb{R}^3`, with the origin at :math:`(1, 0, 0)` and the axes aligned with the global axes.

To create this frame, you can use the following code:

.. code-block:: python

    import py3dframe

    frame = py3dframe.Frame()
    frame.origin([1, 0, 0])

If the frame is not aligned with the global axes, you can set the axis by :
- The rotation matrix
- The Euler angles
- The quaternions
- The axis-angle representation

For example, to set the frame with the Euler angles :math:`(0, \pi/2, 0)`, you can use the following code:

.. code-block:: python

    import py3dframe

    frame = py3dframe.Frame()
    frame.origin([1, 0, 0])
    frame.set_euler_angles([0, math.pi/2, 0], axes='xyz')

When the frame is created, you can convert points and vectors from the global frame to the frame and vice versa.

For example, to convert the point :math:`(1, 1, 1)` from the global frame to the frame, you can use the following code:

.. code-block:: python

    import py3dframe

    frame = py3dframe.Frame()
    frame.origin([1, 0, 0])
    frame.set_euler_angles([0, math.pi/2, 0], axes='xyz')

    point = [1, 1, 1]
    frame_point = frame.from_global_to_frame(point=point)

    print(frame_point)

The result will be:

.. code-block:: console

    [0, 1, -1]

The difference between points and vectors is that the vectors are not translated when converted from the global frame to the frame.

The usefull methods of the frame are:

- from_global_to_frame
- from_frame_to_global
- from_parent_to_frame
- from_frame_to_parent

Operations on frames
--------------------

You can perform operations on frames, such as:
- Inversion

Inversing a frame is equivalent to inverting the transformation matrix of the frame.
Thats means that the new frame space in the old global space and the new global space in the old frame space.

To perform the inversion of a frame, you can use the following code:

.. code-block:: python

    import py3dframe

    frame = py3dframe.Frame()

    inverted_frame = py3dframe.inverse_frame(frame)





