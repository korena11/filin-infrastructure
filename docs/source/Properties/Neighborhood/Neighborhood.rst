Neighborhood
=============

The neighborhood property holds all the neighbors for each point in the point cloud. Each neighborhood is defined by a
:class:`PointNeighborhood.PointNeighborhood`.

Point Neighborhood
-------------------

.. autoclass:: PointNeighborhood.PointNeighborhood
    :members:
    :undoc-members:

    .. rubric:: Attributes

    .. autoautosummary:: PointNeighborhood.PointNeighborhood
        :attributes:

    .. rubric:: Methods

    .. autoautosummary:: PointNeighborhood.PointNeighborhood
        :methods:

Neighbors Property
------------------


.. autoclass:: NeighborProperty.NeighborsProperty
    :members:
    :undoc-members:

    .. rubric:: Attributes

    .. autoautosummary:: NeighborProperty.NeighborsProperty
        :attributes:

    .. rubric:: Methods

    .. autoautosummary:: NeighborProperty.NeighborsProperty
        :methods:


Neighbors Factory
-----------------

.. autoclass:: NeighborsFactory.NeighborsFactory
   :members:
   :undoc-members:

    .. rubric:: Methods

    .. autoautosummary:: NeighborsFactory.NeighborsFactory
        :methods:

.. note::

    The three methods: :meth:`NeighborsFactory.NeighborsFactory.pointSetOpen3D_rnn_kdTree`, :meth:`NeighborsFactory.NeighborsFactory.pointSetOpen3D_knn_kdTree` and :meth:`NeighborsFactory.NeighborsFactory.pointSetOpen3D_krnn_kdTree`
    can be rewritten, without repetition. Only I couldn't be bothered... (RA)
