Usage
==============

This section will guide you through the basic usage of the ``py3dframe`` package.

Construct a rigth-handed frame
------------------------------

First to create a frame, you can give the origin and the axes of the frame as follows:

.. code-block:: python

    import numpy as np
    from py3dframe import Frame

    origin = np.array([1, 2, 3])
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    frame = Frame.FrameTransform(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)


You can also construct a frame from a rotation and a translation using one the 8 possible conventions:

+---------------------+----------------------------------------------------------------+
| Index               | Formula                                                        |
+=====================+================================================================+
| 0                   | :math:`\mathbf{X}_E = \mathbf{R} \mathbf{X}_F + \mathbf{T}`    |
+---------------------+----------------------------------------------------------------+
| 1                   | :math:`\mathbf{X}_E = \mathbf{R} \mathbf{X}_F - \mathbf{T}`    |
+---------------------+----------------------------------------------------------------+
| 2                   | :math:`\mathbf{X}_E = \mathbf{R} (\mathbf{X}_F + \mathbf{T})`  |
+---------------------+----------------------------------------------------------------+
| 3                   | :math:`\mathbf{X}_E = \mathbf{R} (\mathbf{X}_F - \mathbf{T})`  |
+---------------------+----------------------------------------------------------------+
| 4                   | :math:`\mathbf{X}_F = \mathbf{R} \mathbf{X}_E + \mathbf{T}`    |
+---------------------+----------------------------------------------------------------+
| 5                   | :math:`\mathbf{X}_F = \mathbf{R} \mathbf{X}_E - \mathbf{T}`    |
+---------------------+----------------------------------------------------------------+
| 6                   | :math:`\mathbf{X}_F = \mathbf{R} (\mathbf{X}_E + \mathbf{T})`  |
+---------------------+----------------------------------------------------------------+
| 7                   | :math:`\mathbf{X}_F = \mathbf{R} (\mathbf{X}_E - \mathbf{T})`  |
+---------------------+----------------------------------------------------------------+

Where :math:`\mathbf{X}_E` is the point expressed in the parent (or global) frame :math:`E`, :math:`\mathbf{X}_F` is the point expressed in the child (or local) frame :math:`F`, :math:`\mathbf{R}` is the rotation matrix and :math:`\mathbf{T}` is the translation vector.

.. code-block:: python

    from py3dframe import Frame, Rotation

    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    translation = np.array([1, 2, 3]).reshape(3, 1)

    frame = Frame.from_rotation(translation=translation, rotation=rotation, convention=0)

You can check the coordinates of the frame by plotting it into a 3D plot:

.. code-block:: python

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    frame.plot(ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

Construct a system of frames
----------------------------

Lets consider a person walking in a train. 
The person is represented by a frame :math:`F` and the train is represented by a frame :math:`E`.
It is possible to represent the position of the person in the train by defining the frame :math:`F` in the frame :math:`E` coordinates.

.. code-block:: python

    from py3dframe import Frame, FrameTransform

    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    translation = np.array([1, 2, 3]).reshape(3, 1)

    frame_E = Frame.from_rotation(translation=translation, rotation=rotation, convention=0)

    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    translation = np.array([0, 0, 0]).reshape(3, 1)

    frame_F = Frame.from_rotation(translation=translation, rotation=rotation, convention=0, parent=frame_E)

In this case, when the frame :math:`E` moves, the frame :math:`F` moves with it.

Transformation between frames
------------------------------

The transformation between two frames can be computed using the :class:`FrameTransform` class:

.. code-block:: python

    from py3dframe import Frame, FrameTransform

    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    translation = np.array([1, 2, 3]).reshape(3, 1)

    frame_E = Frame.from_rotation(translation=translation, rotation=rotation, convention=0)

    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    translation = np.array([0, 0, 0]).reshape(3, 1)

    frame_F = Frame.from_rotation(translation=translation, rotation=rotation, convention=0, parent=frame_E)

    transform = FrameTransform(input_frame=frame_E, output_frame=frame_F)

    print(transform.translation)
    print(transform.rotation.as_euler('xyz', degrees=True))

This object can be used to transform points or vector from one frame to another:

.. code-block:: python

    point_E = np.array([1, 2, 3]).reshape(3, 1)
    point_F = transform.transform(point=point_E) # In convention 0 : pE = R * pF + T
    point_E = transform.inverse_transform(point=point_F) 

    vector_E = np.array([1, 2, 3]).reshape(3, 1)
    vector_F = transform.transform(vector=vector_E) # In convention 0 : vE = R * vF
    vector_E = transform.inverse_transform(vector=vector_F)

When the frame :math:`E` moves, the transform object will automatically update the transformation between the two frames.