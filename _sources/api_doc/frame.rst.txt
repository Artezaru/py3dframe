.. currentmodule:: py3dframe

py3dframe.Frame
===============

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top


Frame class
-------------------------------------------

.. autoclass:: Frame


Define the frame relatively to a parent frame
---------------------------------------------------------------------------------------

A frame of reference can be defined relatively to another frame of reference called the ``parent`` frame.
If no parent frame is provided, the frame is defined relative to the global frame (the canonical frame of reference of :math:`\mathbb{R}^3`).

For example, a person in a train can be represented by a frame called `person_frame` defined relatively to the frame of the train called `train_frame`.
When the train moves, the `train_frame` is updated. The `person_frame` remains the same relatively to the `train_frame` but its position and orientation in the global frame changes.

.. figure:: ../../../py3dframe/resources/train_person_frame.png

The transformation between the frame and its parent frame is defined by a rotation and a translation in a given convention.
You can access the convention, the parent frame and extract the representation of the frame in the global frame coordinates system using the properties and methods below.

.. autosummary::
   :toctree: ../generated/

    Frame.convention
    Frame.parent
    Frame.get_global_frame


Create and manipulate Frames with axes and origin
-----------------------------------------------------------------------

A Frame object can be manipulated by its axes and origin.

To define a frame using its axes and origin, you can use the following class method:

.. autosummary::
   :toctree: ../generated/

   Frame.canonical
   Frame.from_axes

Once the frame is created, you can access and set the axes and origin of the current frame expressed in the parent frame coordinates system.

.. autosummary::
   :toctree: ../generated/

    Frame.origin
    Frame.axes
    Frame.x_axis
    Frame.y_axis
    Frame.z_axis

To extract the axes and origin of the current frame expressed in the global frame coordinates system, you can use the properties and methods below.
If the current frame has no parent, the global quantities are equal to the local quantities defined above.

.. autosummary::
   :toctree: ../generated/

    Frame.global_origin
    Frame.global_axes
    Frame.global_x_axis
    Frame.global_y_axis
    Frame.global_z_axis


Create and manipulate Frames with rotation and translation
----------------------------------------------------------------------

General transformation conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Frame object can be manipulated by its rotation and translation expressed in a given convention.
The construction of the rotation and translation depends on the convention used.

Lets note :math:`E` the parent frame (or the global frame) and :math:`F` the frame to define.
The transformation between the frame :math:`E` and the frame :math:`F` is defined by a rotation matrix :math:`R` and a translation vector :math:`T`.
Several conventions can be used to define the transformation between the frames.

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

The default convention is 0.

To define a frame using its rotation and translation, you can use the following class methods:

.. autosummary::
   :toctree: ../generated/

   Frame.from_rotation

Once the frame is created, you can access and set the rotation and translation of the current frame expressed in the parent frame coordinates system.

.. autosummary::
   :toctree: ../generated/

    Frame.translation
    Frame.get_translation
    Frame.set_translation
    Frame.rotation
    Frame.get_rotation
    Frame.set_rotation

To extract the rotation and translation of the current frame expressed in the global frame coordinates system, you can use the properties and methods below.
If the current frame has no parent, the global quantities are equal to the local quantities defined above.

.. autosummary::
   :toctree: ../generated/

    Frame.global_translation
    Frame.get_global_translation
    Frame.set_global_translation
    Frame.global_rotation
    Frame.get_global_rotation
    Frame.set_global_rotation


Rotation matrix
~~~~~~~~~~~~~~~~~~~~

If you want to define the rotation using a rotation matrix instead of a scipy Rotation object, you can use the following class method:

.. autosummary::
   :toctree: ../generated/

   Frame.from_rotation_matrix

The quantities related to the rotation matrix can be accessed and set using the methods and properties below.

.. autosummary::
   :toctree: ../generated/

    Frame.rotation_matrix
    Frame.get_rotation_matrix
    Frame.set_rotation_matrix
    Frame.global_rotation_matrix
    Frame.get_global_rotation_matrix
    Frame.set_global_rotation_matrix


Quaternion
~~~~~~~~~~~~~~~~~~~~

If you want to define the rotation using a quaternion instead of a scipy Rotation object, you can use the following class method:

.. autosummary::
   :toctree: ../generated/

   Frame.from_quaternion

The quantities related to the quaternion can be accessed and set using the methods and properties below.

.. autosummary::
   :toctree: ../generated/

    Frame.quaternion
    Frame.get_quaternion
    Frame.set_quaternion
    Frame.global_quaternion
    Frame.get_global_quaternion
    Frame.set_global_quaternion


Euler angles
~~~~~~~~~~~~~~~~~~~~

If you want to define the rotation using Euler angles instead of a scipy Rotation object, you can use the following class method:

.. autosummary::
   :toctree: ../generated/

   Frame.from_euler_angles

The quantities related to the Euler angles can be accessed and set using the methods and properties below.

.. autosummary::
   :toctree: ../generated/

    Frame.euler_angles
    Frame.get_euler_angles
    Frame.set_euler_angles
    Frame.global_euler_angles
    Frame.get_global_euler_angles
    Frame.set_global_euler_angles


Rotation vector
~~~~~~~~~~~~~~~~~~~~

If you want to define the rotation using a rotation vector instead of a scipy Rotation object, you can use the following class method:

.. autosummary::
   :toctree: ../generated/

   Frame.from_rotation_vector

The quantities related to the rotation vector can be accessed and set using the methods and properties below.

.. autosummary::
   :toctree: ../generated/

    Frame.rotation_vector
    Frame.get_rotation_vector
    Frame.set_rotation_vector
    Frame.global_rotation_vector
    Frame.get_global_rotation_vector
    Frame.set_global_rotation_vector


Save, load and other manipulation of Frames
-------------------------------------------

A Frame object can be saved and loaded using the methods below.

.. autosummary::
   :toctree: ../generated/

    Frame.save_to_dict
    Frame.load_from_dict
    Frame.save_to_json
    Frame.load_from_json

A Frame object can be copied using the method below.

.. autosummary::
   :toctree: ../generated/

    Frame.copy
    Frame.deepcopy

Two frames can be compared using ``==`` and ``!=`` operators. The comparison is done by comparing the origin and the basis vectors of the frames in the global frame coordinates system.

Examples of Frame creation and manipulation
-------------------------------------------

Lets :math:`E = (O_E, \mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3)` be a frame of reference of :math:`\mathbb{R}^3`.
To create the frame :math:`E`, the user can provide the origin and the basis vectors of the frame.

.. code-block:: python

   from py3dframe import Frame

   origin = [1, 2, 3]
   x_axis = [1, 1, 0]
   y_axis = [1, -1, 0]
   z_axis = [0, 0, 1]
   frame_E = Frame.from_axes(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, parent=None)

A frame can also be defined relatively to another frame.

Lets consider a frame :math:`F = (O_F, \mathbf{f}_1, \mathbf{f}_2, \mathbf{f}_3)` defined in the frame :math:`E`.
The user must provide the origin and the basis vectors of the frame :math:`F` in the frame :math:`E`.

.. code-block:: python

   from py3dframe import Frame

   origin = [1, -2, 3]
   x_axis = [1, 0, 1]
   y_axis = [0, 1, 0]
   z_axis = [-1, 0, 1]
   frame_F = Frame.from_axes(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, parent=frame_E)

In this case the frame :math:`F` is defined relatively to the frame :math:`E`.
A change in the frame :math:`E` will affect the frame :math:`F` but the transformation between the frames will remain the same.

The user can access the origin and the basis vectors of the frame as follows:

.. code-block:: python

   # In the global frame (canonical frame of R3) 
   # -> changed by a move of the parent frame
   frame_F.global_origin 
   frame_F.global_x_axis
   frame_F.global_y_axis
   frame_F.global_z_axis

   # In the parent frame coordinates and relative to the parent frame 
   # -> unchanged by a move of the parent frame
   frame_F.origin
   frame_F.x_axis
   frame_F.y_axis
   frame_F.z_axis

To finish, the user can define a frame using the transformation between the parent frame and the frame.
Using the convention 0, the rotation matrix and the translation vector will exactly be the basis vectors and the origin of the frame.

.. code-block:: python

   from py3dframe import Frame

   translation = [1, 2, 3]
   rotation_matrix = np.array([[1, 1, 0], [1, -1, 0], [0, 0, 1]]).T # Equivalent to the column_stack((x_axis, y_axis, z_axis))
   rotation_matrix = rotation_matrix / np.linalg.norm(rotation_matrix, axis=0) # Normalize the columns to get an orthonormal matrix
   frame_E = Frame.from_rotation_matrix(translation=translation, rotation_matrix=rotation_matrix, parent=None, convention=0)
