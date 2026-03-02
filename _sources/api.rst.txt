API Reference
==============

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top


Main classes
-------------------------------

The package ``py3dframe`` is composed of the following main classes:

- :doc:`Rotation <_docs/rotation>` class is used to manage 3D rotations (alias of ``scipy.spatial.transform.Rotation``).
- :doc:`Frame <_docs/frame>` class is used to represent 3D frames of reference.
- :doc:`FrameTree <_docs/frame_tree>` class is used to manage hierarchical relationships between multiple frames (easier than using :attr:`py3dframe.Frame.parent` attributes).
- :doc:`FrameTransform <_docs/frame_transform>` class is used to manage 3D transformations between frames.


.. toctree::
   :maxdepth: 1
   :hidden:

   ./_docs/rotation.rst
   ./_docs/frame.rst
   ./_docs/frame_tree.rst
   ./_docs/frame_transform.rst


Frame manipulation functions
------------------------------

Some manipulation functions for :class:`py3dframe.Frame` objects are provided to easily create new frames from existing ones:

- :doc:`Rotation around an axis <_docs/rotate_around_axis>` function is used to create a new frame by rotating an existing frame around a specified axis.
- :doc:`Symmetry across a plane <_docs/mirror_across_plane>` function is used to create a new frame by mirroring an existing frame across a specified plane.
- :doc:`Translation <_docs/translate>` function is used to create a new frame by translating an existing frame by a specified translation vector.
- :doc:`Translation along an axis <_docs/translate_along_axis>` function is used to create a new frame by translating an existing frame along a specified axis by a specified distance.
 

.. toctree::
   :maxdepth: 1
   :hidden:

   ./_docs/mirror_across_plane.rst
   ./_docs/rotate_around_axis.rst
   ./_docs/translate.rst
   ./_docs/translate_along_axis.rst


Matrix submodule
------------------------------

Some additional utility functions are also provided in the :mod:`py3dframe.matrix` module in order to manipulate 3D matrices in :math:`O(3)` and :math:`SO(3)` groups:

.. toctree::
   :maxdepth: 1
   :caption: matrix submodule:

   ./_docs/is_O3.rst
   ./_docs/is_SO3.rst
   ./_docs/O3_project.rst
   ./_docs/SO3_project.rst


Frame convention conversion
--------------------------------

Finally, to perform conversions between the different conventions used in the literature for representing 3D rotations and transformations, a function :func:`py3dframe.switch_RT_convention` is provided:

.. toctree::
   :maxdepth: 1
   :caption: Conversion functions:

   ./_docs/switch_RT_convention.rst

