Welcome to Filin Students' infrastucture documentation!
========================================================


The infrastucture is object oriented. There are three data-set object ( :class:`~PointSet.PointSet`, :class:`~PointSubSet.PointSubSet` and :class:`~RasterData.RasterData` ) which have a basic set of properties, Factoy objects and Property objects. 

The Property objects relate to the data-set, and are built by a Factory object. In each Factory there several functions that build the same property, but with a different algorithm. 

Visualization is applicable through Visualization Classes. 

Other general utilities are in :py:mod:`~MyTools`.

Contents:
=========

.. toctree::

    Datasets/PointSet
    Datasets/RasterData
    Properties/BaseProperty
    Factories/Factories
    LevelSets/LevelSets
    Utils/EigenFactory
    Utils/Utils
    Visualizations/Visualization
	   
Dependencies
============
This project is implemented for Python 3.6 

- `pyProj <https://jswhit.github.io/pyproj/>`_.
   Installation via `pip install pyproj`
- `vtk <https://www.vtk.org/>`_
   Installation via `pip install vtk`
- `opencv <https://opencv.org/>`_
   Installation via `pip install opencv-python`
- `geopandas <http://geopandas.org/>`_
   Installation via `conda install -c conda-forge geopandas`
- `gdal <https://www.gdal.org/>`_
   Installation via `pip install GDAL`
- `pyspark <https://spark.apache.org/docs/0.9.0/python-programming-guide.html>`_
   Make sure you have Java 8 or higher installed on your computer.

   Installation via `pip install pyspark`

References
----------

.. bibliography:: refs1.bib
   :style: plain
   :cited:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



