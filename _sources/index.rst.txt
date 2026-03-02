Welcome to py3dframe's documentation!
=====================================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top


Description of the package
--------------------------

``py3dframe`` provides tools to create, manipulate and query **orthogonal,
right‑handed 3‑D frames of reference**.  

.. note::

   The package is designed to work with double-precision floating-point numbers to ensure numerical stability in all calculations.
   Therefore, all float arrays are automatically converted to ``numpy.float64`` for computation and all integer arrays are converted to ``numpy.int64`` for computation.
   This means that when you pass arrays to the functions in the package, they will be converted to these data types if they are not already in that format.

Contents
--------

.. grid:: 3

    .. grid-item-card:: 
      :img-top: /_static/_icons/download.png
      :text-align: center

      Installation
      ^^^

      This section describes how to install the package into a Python environment. 
      It includes instructions for installing the package using pip, as well as any necessary dependencies.

      +++

      .. button-ref:: installation
         :expand:
         :color: secondary
         :click-parent:

         To the installation guide

    .. grid-item-card::
      :img-top: /_static/_icons/api.png
      :text-align: center

      API Reference
      ^^^

      The reference guide contains a detailed description of the functions,
      modules, and objects included in ``py3dframe``. The reference describes how the
      methods work and which parameters can be used.

      +++ 

      .. button-ref:: api
         :expand:
         :color: secondary
         :click-parent:

         To the API reference

    .. grid-item-card::
      :img-top: /_static/_icons/examples.png
      :text-align: center

      Examples Gallery
      ^^^

      This section contains a collection of examples demonstrating how to use the package for various applications. 
      Each example includes a description of the problem being solved, the code used to solve it, and the resulting output.

      +++

      .. button-ref:: _gallery/index
         :expand:
         :color: secondary
         :click-parent:

         To the examples gallery

.. toctree::
   :caption: Contents:
   :hidden:

   installation
   api
   _gallery/index


Basic usage
------------------------------

First to create a frame, you can give the origin and the axes of the frame as follows:

.. code-block:: python

    import numpy as np
    from py3dframe import Frame

    origin = np.array([1, 2, 3])
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    frame = Frame.from_axes(origin=origin, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)


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
    

Author
------

The package ``py3dframe`` was created by the following authors:

- Artezaru <artezaru.github@proton.me>

You can access the package and the documentation with the following URL:

- **Git Plateform**: https://github.com/Artezaru/py3dframe.git
- **Online Documentation**: https://Artezaru.github.io/py3dframe

License
-------

Please refer to the [LICENSE] file for the license of the package.
