Data set classes
=================

Each dataset class inherits from the :class:`BaseData.BaseData` class (except for :class:`CurveData.CurveData`). Some of the functions work with a specific dataset class, which
sometimes inherits from :class:`PointSet.PointSet`, and add more functionality (such as ball tree or the open3D
PointCloud object).

The following are the existing dataset classes:

.. toctree::
    :maxdepth: 2

    BaseData
    PointSet
    PointSetOpen3D
    BallTreePointSet
    RasterData
    CurveData

