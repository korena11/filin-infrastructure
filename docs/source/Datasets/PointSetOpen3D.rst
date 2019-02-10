PointSetOpen3D class
====================

This class inherits from :class:`PointSet.PointSet` and adds functionality (such as normals computation) via
:ref:`Open3D <open3d:tutorial>`.

The class computes a KD-tree with initialization. As used in :ref:`Open3D <open3d:KDTree>`, it uses `FLANN <https://www.cs.ubc.ca/research/flann/>`_ to build KDTrees for fast retrieval of nearest neighbors.

When a subset of the PointSet is required, a subclass is set use :ref:`PointSubSetOpen3D class`


.. autoclass:: PointSetOpen3D.PointSetOpen3D
    :members:
    :undoc-members:

    .. rubric:: Attributes

    .. autoautosummary:: PointSetOpen3D.PointSetOpen3D
        :attributes:

    .. rubric:: Methods

    .. autoautosummary:: PointSetOpen3D.PointSetOpen3D
        :methods:

PointSubSetOpen3D class
------------------------
.. autoclass:: PointSubSetOpen3D.PointSubSetOpen3D
    :members:
    :undoc-members:

    .. rubric:: Attributes

    .. autoautosummary:: PointSubSetOpen3D.PointSubSetOpen3D
        :attributes:

    .. rubric:: Methods

    .. autoautosummary:: PointSubSetOpen3D.PointSubSetOpen3D
        :methods: