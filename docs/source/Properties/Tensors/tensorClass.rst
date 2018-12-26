Tensors
=======

The tensor property holds a tensor for a point or a segment. Each tensor is defined by a
:class:`tensor.Tensor`. The :class:`TensorProperty.TensorProperty` holds all the tensors for a point-cloud.
The :class:`TensorFactory.TensorFactory` builds either a `tensor` object or a `TensorProperty`.


.. toctree::

    tensorProperty
    TensorFactory


tensor class
------------

.. autoclass:: tensor.Tensor
    :members:
    :undoc-members:

    .. rubric:: Attributes

    .. autoautosummary:: tensor.Tensor
        :attributes:

    .. rubric:: Methods

    .. autoautosummary:: tensor.Tensor
        :methods:


Tensor Property
---------------


.. autoclass:: TensorProperty.TensorProperty
    :members:
    :undoc-members:

    .. rubric:: Attributes

    .. autoautosummary:: TensorProperty.TensorProperty
        :attributes:

    .. rubric:: Methods

    .. autoautosummary:: TensorProperty.TensorProperty
        :methods:


Tensor Factory
---------------


.. autoclass:: TensorFactory.TensorFactory
   :members:
   :undoc-members:

    .. rubric:: Methods

    .. autoautosummary:: TensorFactory.TensorFactory
        :methods: