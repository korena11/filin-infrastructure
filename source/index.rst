
Welcome to Filin Students' infrastucture documentation!
========================================================


The infrastucture is object oriented. There are three data-set object ( :class:`~PointSet.PointSet`, :class:`~PointSubSet.PointSubSet` and :class:`~RasterData.RasterData` ) which have a basic set of properties, Factoy objects and Property objects. 

The Property objects relate to the data-set, and are built by a Factory object. In each Factory there several functions that build the same property, but with a different algorithm. 

Visualization is applicable through Visualization Classes. 

Contents:
=========

.. toctree::
	:maxdepth: 3

	PointSet

	RasterData

	BaseProperty

	Factories
	   
	   
Dependencies
============
This project is implemented for Python 3.6 

- pyProj
- vtk
- opencv
- geopandas


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



