API Reference
==============

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top


Overview of the py3dframe package
---------------------------------

Main classes
~~~~~~~~~~~~~~~~~~~

The package ``py3dframe`` is composed of the following main classes:

- :class:`py3dframe.Rotation` class is used to manage 3D rotations (alias of ``scipy.spatial.transform.Rotation``).
- :class:`py3dframe.Frame` class is used to represent 3D frames of reference.
- :class:`py3dframe.FrameTree` class is used to manage hierarchical relationships between multiple frames (easier than using :attr:`py3dframe.Frame.parent` attributes).
- :class:`py3dframe.FrameTransform` class is used to manage 3D transformations between frames.

.. toctree::
   :maxdepth: 1
   :caption: Main classes:

   ./api_doc/rotation.rst
   ./api_doc/frame.rst
   ./api_doc/frame_tree.rst
   ./api_doc/frame_transform.rst


Manipulate frames and transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some manipulation functions for :class:`py3dframe.Frame` objects are provided to easily create new frames from existing ones:
 
.. toctree::
   :maxdepth: 1
   :caption: Frame manipulation functions:

   ./api_doc/mirror_across_plane.rst
   ./api_doc/rotate_around_axis.rst
   ./api_doc/translate.rst
   ./api_doc/translate_along_axis.rst


Matrix submodule
------------------------------

Some additional utility functions are also provided in the :mod:`py3dframe.matrix` module in order to manipulate 3D matrices in :math:`O(3)` and :math:`SO(3)` groups:

.. toctree::
   :maxdepth: 1
   :caption: matrix submodule:

   ./api_doc/is_O3.rst
   ./api_doc/is_SO3.rst
   ./api_doc/O3_project.rst
   ./api_doc/SO3_project.rst


Frame convention conversion
--------------------------------

Finally, to perform conversions between the different conventions used in the literature for representing 3D rotations and transformations, a function :func:`py3dframe.switch_RT_convention` is provided:

.. toctree::
   :maxdepth: 1
   :caption: Conversion functions:

   ./api_doc/switch_RT_convention.rst

To learn how to use the package effectively, refer to the documentation :doc:`../usage`.