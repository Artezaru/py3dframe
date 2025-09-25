API Reference
==============

The package ``py3dframe`` is composed of the following main classes:

- :class:`py3dframe.Rotation` class is used to manage 3D rotations (alias of ``scipy.spatial.transform.Rotation``).
- :class:`py3dframe.Frame` class is used to manage 3D frames.
- :class:`py3dframe.FrameTransform` class is used to manage 3D transformations between frames.

.. toctree::
   :maxdepth: 1
   :caption: Main classes:

   ./api_doc/rotation.rst
   ./api_doc/frame.rst
   ./api_doc/frame_transform.rst

Some additional utility functions are also provided in the :mod:`py3dframe.matrix` module in order to manipulate 3D matrices in :math:`O(3)` and :math:`SO(3)` groups:

.. toctree::
   :maxdepth: 1
   :caption: matrix submodule:

   ./api_doc/is_O3.rst
   ./api_doc/is_SO3.rst
   ./api_doc/O3_project.rst
   ./api_doc/SO3_project.rst

Finally, to perform conversions between the different conventions used in the literature for representing 3D rotations and transformations, a function :func:`py3dframe.switch_RT_convention` is provided:

.. toctree::
   :maxdepth: 1
   :caption: Conversion functions:

   ./api_doc/switch_RT_convention.rst

To learn how to use the package effectively, refer to the documentation :doc:`../usage`.