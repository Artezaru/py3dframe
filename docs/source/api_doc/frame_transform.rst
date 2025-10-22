.. currentmodule:: py3dframe

py3dframe.FrameTransform
=========================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

.. autoclass:: FrameTransform

Set the input and output frames
--------------------------------

The input and output frames of the transformation can be set and accessed via the corresponding properties.
To freeze the transformation with respect to changes in the input and output frames, set the ``dynamic`` attribute to False.

.. autosummary::
   :toctree: ../generated/

   FrameTransform.input_frame
   FrameTransform.output_frame
   FrameTransform.dynamic
   FrameTransform.convention
   FrameTransform.get_active_input_frame
   FrameTransform.get_active_output_frame

Access the parameters of the transformation
--------------------------------------------

.. autosummary::
   :toctree: ../generated/

   FrameTransform.get_translation
   FrameTransform.translation
   FrameTransform.get_rotation
   FrameTransform.rotation
   FrameTransform.get_rotation_matrix
   FrameTransform.rotation_matrix
   FrameTransform.get_quaternion
   FrameTransform.quaternion
   FrameTransform.get_euler_angles
   FrameTransform.euler_angles
   FrameTransform.get_rotation_vector
   FrameTransform.rotation_vector

Perform the transformation from input to output and vice versa
-----------------------------------------------------------------

To transform points or vectors from the input frame to the output frame, use the ``transform`` method.
To transform points or vectors from the output frame to the input frame, use the ``inverse_transform`` method.

.. autosummary::
   :toctree: ../generated/

   FrameTransform.transform
   FrameTransform.inverse_transform

Manipulate the transformation
-------------------------------

Some operations can be performed on FrameTransform objects.

.. autosummary::
    :toctree: ../generated/

    FrameTransform.inverse


Examples of Usage
--------------------

To create a FrameTransform object, the user must provide two Frame objects.

.. code-block:: python

    from py3dframe import Frame, FrameTransform

    frame_E = Frame.canonical() # Global frame
    frame_F = Frame.from_axes(origin=[1, 2, 3], x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1]) # Local frame
    transform = FrameTransform(input_frame=frame_E, output_frame=frame_F, dynamic=True, convention=0)

- If the ``dynamic`` parameter is set to ``True``, the FrameTransform object will be affected by the changes in the input frame or the output frame.
- If the ``dynamic`` parameter is set to ``False``, the FrameTransform object will correspond to the transformation between the input frame and the output frame at the time of the creation of the FrameTransform object.
- If the ``dynamic`` parameter is set to ``True`` and then changed to ``False``, the FrameTransform object will correspond to the transformation between the input frame and the output frame at the time of the change of the ``dynamic`` parameter.

The user can access the rotation matrix and the translation vector of the transformation as follows:

.. code-block:: python

    R = transform.get_rotation_matrix(convention=0) # Rotation matrix in convention 0
    T = transform.get_translation(convention=0) # Translation vector in convention 0

The user can also access the input frame and the output frame of the transformation as follows:

.. code-block:: python

    frame_E = transform.get_active_input_frame()
    frame_F = transform.get_active_output_frame()

To update the input frame or the output frame of the transformation, the user can use the corresponding properties.

.. code-block:: python

    frame_G = Frame.from_axes(origin=[3, 2, 1], x_axis=[0, 1, 0], y_axis=[0, 0, 1], z_axis=[1, 0, 0]) # Another local frame
    transform.input_frame = frame_G # Update the input frame of the transformation
    transform.output_frame = frame_F # Update the output frame of the transformation

.. note::

    If the ``dynamic`` parameter is set to ``False``, the FrameTransform object will not be affected by the changes in the input frame or the output frame.
    Set ``dynamic`` to ``True`` to reactivate the dynamic mode (you can then reset it to ``False`` again if needed).

The FrameTransform object can be used to transform points or vectors from the input frame to the output frame and vice versa.

.. code-block:: python

    X_i = [1, 2, 3]
    X_o = transform.transform(point=X_i)
    X_i = transform.inverse_transform(point=X_o)

For vectors, the translation vector is not taken into account.

.. code-block:: python

    X_i = [1, 2, 3]
    X_o = transform.transform(point=X_i) # X_i = R X_o + T
    V_i = [1, 2, 3]
    V_o = transform.transform(vector=V_i) # V_i = R V_o