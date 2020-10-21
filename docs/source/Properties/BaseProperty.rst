Properties
===========

Each property inherits from this class. If values need to be set, each property has its own :func:`~BaseProperty.setValues` , and it should be defined for new properties. 
The basic methods and attributes for every property are:

.. rubric:: Attributes

    .. autoautosummary:: BaseProperty.BaseProperty
        :attributes:


    .. rubric:: Methods

    .. autoautosummary:: BaseProperty.BaseProperty
        :methods:

    .. rubric:: Methods documentation

    .. autoclass:: BaseProperty.BaseProperty
        :members:
        :undoc-members:

Each property is created by a Factory. These are the existing properties and factories:

.. toctree::

    Colors/Colors
    Curvature/Curvature
    EigenProperty
    Normals/Normals
    Neighborhood/Neighborhood
    Panorama/Panorama
    Saliency/Saliency
    SegmentationProperty
    SphericalCoordinates/Spherical
    Transformations/Transformation
    TriangulationProperty
    Tensors/tensorClass

