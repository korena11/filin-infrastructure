Data set classes
=================

Each dataset class inherits from the `BaseData` class (except for `CurveData`). Some of the functions work with a specific dataset class, which
sometimes inherits from :class:`PointSet.PointSet`, and add more functionality (such as ball tree or the open3D
PointCloud object).

The following are the existing dataset classes:

.. toctree::
    :maxdepth: 2

    PointSet
    PointSetOpen3D
    RasterData
    CurveData

