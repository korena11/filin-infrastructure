
Welcome to Filin Students' infrastucture documentation!
========================================================


The infrastucture is object oriented. There are three data-set object ( :class:`~PointSet.PointSet`, :class:`~PointSubSet.PointSubSet` and :class:`~RasterData.RasterData` ) which have a basic set of properties, Factoy objects and Property objects. 

The Property objects relate to the data-set, and are built by a Factory object. In each Factory there several functions that build the same property, but with a different algorithm. 

Visualization is applicable through Visualization Classes. 

Contents:
=========

.. toctree::

    Datasets/PointSet
    Datasets/RasterData
    Properties/BaseProperty
    Factories/Factories
    LevelSets/LevelSets
    Utils/EigenFactory
    Visualizations/Visualization
	   
Dependencies
============
This project is implemented for Python 3.6 

- `pyProj <https://pypi.org/project/pyproj/>`.
   Installation via `pip install pyproj`
- `vtk <https://pypi.org/project/vtk/>`
   Installation via `pip install vtk`
- `opencv <https://pypi.org/project/opencv-python/>`
   Installation via `pip install opencv-python`
- `geopandas <http://geopandas.org/>`
   Installation via `conda install -c conda-forge geopandas`
- `gdal <https://pypi.org/project/GDAL/>`
   Installation via `pip install GDAL`
- `pyspark <https://pypi.org/project/pyspark/>` make sure you have Java 8 or higher installed on your computer.
   Installation via `pip install pyspark`


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



