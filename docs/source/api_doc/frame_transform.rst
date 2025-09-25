.. currentmodule:: py3dframe

py3dframe.FrameTransform
=========================

.. autoclass:: FrameTransform

Set the input and output frames
--------------------------------

.. autosummary::
   :toctree: Transform_generated/

   FrameTransform.input_frame
   FrameTransform.output_frame
   FrameTransform.dynamic
   FrameTransform.convention

Access the parameters of the transformation
--------------------------------------------

.. autosummary::
   :toctree: Transform_generated/

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

Perform the transformation
--------------------------------

.. autosummary::
   :toctree: Transform_generated/

   FrameTransform.transform
   FrameTransform.inverse_transform