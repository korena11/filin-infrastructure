Neighborhood
=============

The neighborhood property holds all the neighbors for each point in the point cloud. Each neighborhood is defined by a
:class:`PointNeighborhood.PointNeighborhood`.

Point Neighborhood
-------------------

.. rubric:: Attributes

.. autoautosummary:: PointNeighborhood.PointNeighborhood
   :attributes:

.. rubric:: Methods

.. autoautosummary:: PointNeighborhood.PointNeighborhood
    :methods:

.. autoclass:: PointNeighborhood.PointNeighborhood
    :members:
    :undoc-members:

Neighbors Property
------------------
.. rubric:: Attributes

.. autoautosummary:: NeighborsProperty.NeighborsProperty
    :attributes:

.. autoclass:: NeighborsProperty.NeighborsProperty
    :members:
    :undoc-members:



.. rubric:: Methods

.. autoautosummary:: NeighborsProperty.NeighborsProperty
    :methods:


Neighbors Factory
-----------------

.. rubric:: Methods

.. autoautosummary::
    NeighborsFactory.NeighborsFactory
    :methods:

.. autoclass:: NeighborsFactory.NeighborsFactory
   :members:
   :undoc-members:

.. note::

    The three methods: :meth:`NeighborsFactory.NeighborsFactory.pointSetOpen3D_rnn_kdTree`, :meth:`NeighborsFactory.NeighborsFactory.pointSetOpen3D_knn_kdTree` and :meth:`NeighborsFactory.NeighborsFactory.pointSetOpen3D_krnn_kdTree`
    can be rewritten, without repetition. Only I couldn't be bothered... (RA)
