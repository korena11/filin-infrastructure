VisualizationO3D Documentation
==============================

Visualization class that uses Open3D for showing point clouds and their properties

The VisualizationO3D can work both with PointSet and BaseProperty.

- **'K'**: pressing this button the background will change to black

PointSet visualization
''''''''''''''''''''''

.. literalinclude:: ../../../Visualization Classes/test_visualizationO3D.py
   :language: python
   :lineno-start: 13
   :lines: 13-19
   :linenos:

This function gets a PointSet and a ColorProperty, if exists, and shows the point cloud with the original colors.

BaseProperty visualization
''''''''''''''''''''''''''

.. literalinclude:: ../../../Visualization Classes/test_visualizationO3D.py
   :language: python
   :lineno-start: 22
   :lines: 22-33
   :linenos:

Here two new keyboard functions added:

- **'P'**: changes the attribute shown from the property (every attribute that is not private and that is described by a numpy array.
- **'.'**: changes the colormap of the attribute shown.

Both keys are cyclic, so every press will change the colormap or the attribute.

.. note::

    Other colormaps can be added, according the definition of matplotlib colormaps.

.. autoclass:: VisualizationO3D.VisualizationO3D
   :members:
   :undoc-members:

    .. rubric:: Attributes

    .. autoautosummary:: VisualizationO3D.VisualizationO3D
        :attributes:

    .. rubric:: Methods

    .. autoautosummary:: VisualizationO3D.VisualizationO3D
        :methods:
