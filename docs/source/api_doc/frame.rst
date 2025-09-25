.. currentmodule:: py3dframe

py3dframe.Frame
===============

.. autoclass:: Frame

Instanciation of a Frame
-------------------------

To create a Frame, you can use the following constructors:

.. autosummary::
   :toctree: Frame_generated/

   Frame.canonical
   Frame.from_rotation
   Frame.from_axes
   Frame.from_rotation_matrix
   Frame.from_quaternion
   Frame.from_euler_angles
   Frame.from_rotation_vector

Access, Set and Get Frame Properties
-------------------------------------

The transformation properties are relative to a convention and a parent frame.
To set up this parameters, you can use the following methods and properties:

.. autosummary::
   :toctree: Frame_generated/

    Frame.parent
    Frame.convention

The transformation properties can be accessed, set and get using the methods and properties below.
This includes local transformation properties (relative to the parent frame)

.. autosummary::
   :toctree: Frame_generated/

    Frame.origin
    Frame.axes
    Frame.x_axis
    Frame.y_axis
    Frame.z_axis
    Frame.translation
    Frame.get_translation
    Frame.set_translation
    Frame.rotation
    Frame.get_rotation
    Frame.set_rotation
    Frame.rotation_matrix
    Frame.get_rotation_matrix
    Frame.set_rotation_matrix
    Frame.quaternion
    Frame.get_quaternion
    Frame.set_quaternion
    Frame.euler_angles
    Frame.get_euler_angles
    Frame.set_euler_angles
    Frame.rotation_vector
    Frame.get_rotation_vector
    Frame.set_rotation_vector

The global transformation properties (relative to the canonical frame) can be accessed, set and get using the methods and properties below.

.. autosummary::
   :toctree: Frame_generated/

    Frame.get_global_frame
    Frame.global_origin
    Frame.global_axes
    Frame.global_x_axis
    Frame.global_y_axis
    Frame.global_z_axis
    Frame.global_translation
    Frame.get_global_translation
    Frame.set_global_translation
    Frame.global_rotation
    Frame.get_global_rotation
    Frame.set_global_rotation
    Frame.global_rotation_matrix
    Frame.get_global_rotation_matrix
    Frame.set_global_rotation_matrix
    Frame.global_quaternion
    Frame.get_global_quaternion
    Frame.set_global_quaternion
    Frame.global_euler_angles
    Frame.get_global_euler_angles
    Frame.set_global_euler_angles
    Frame.global_rotation_vector
    Frame.get_global_rotation_vector
    Frame.set_global_rotation_vector

Some additional methods are provided to manipulate frames:

.. autosummary::
   :toctree: Frame_generated/

    Frame.save_to_dict
    Frame.load_from_dict
    Frame.save_to_json
    Frame.load_from_json
