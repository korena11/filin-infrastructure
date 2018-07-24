Level Set functions and factory
================================
In order to get the Level Set going, one must first initialize the LevelSet object through
:class:`~LevelSetFactory.LevelSetFactory`. However, as the factory moves a level set function, according to which the
zero-set is extracted, a LevelSetFunction (:class:`~LevelSetFunction.LevelSetFunction`) is initialized. This function
has its own initlizations and properties which should be addressed when the LevelSetFactory is set to go.

Part of the initialization requires saliency computation, which is made using the class :class:`~Saliency.Saliency`

.. toctree::

    LevelSetFactory
    LevelSetFunction
    Saliency

