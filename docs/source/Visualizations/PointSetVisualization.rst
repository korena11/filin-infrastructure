Visualization Documentation
=====================================

    Visualization class.

    In order to show PointSet a VisualizationVTK object is created

    .. code-block:: py

        vis_obj = VisualizationVTK(number_of_viewports=1)

    Giving the render window a name

    .. code-block:: py

         vis_obj.SetRenderWindowName("Test - One Viewport. PointSet. Uniform Color.")

    Inserting data into the rendering window:

    .. code-block:: py

        vis_obj.Bind(input_data=pointset1, renderFlag='color', color=(255, 0, 0), new_layer=False)

    **Examples**

    One viewport, binding two Pointsets

    .. code-block:: py

        vis_obj = VisualizationVTK(number_of_viewports=1)
        vis_obj.SetRenderWindowName("Test - One Viewport. PointSet. Uniform Color.")
        vis_obj.Bind(pointset1, renderFlag='color', color=(255, 0, 0), new_layer=False)
        vis_obj.Bind(pointset2, renderFlag='color', color=(0, 255, 0), new_layer=True)

    Two viewports, binding two halves with two Pointsets in each half.

    .. code-block:: py

        vis_obj = VisualizationVTK(number_of_viewports=2, two_viewports_vertical_horizontal='V')
        vis_obj.SetRenderWindowName("Test - Two Viewport. PointSet and PointSubSet. Uniform Color.")
        vis_obj.BindFirstHalf(pointset1, renderFlag='color', color=(255, 0, 0), new_layer=False)

        vis_obj.BindFirstHalf(pointset2, renderFlag='color', color=(0, 255, 0), new_layer=True)
        vis_obj.BindSecondHalf(pointsubset1, renderFlag='color', color=(255, 0, 0), new_layer=False)
        vis_obj.BindSecondHalf(pointsubset2, renderFlag='color', color=(0, 255, 0), new_layer=True)

    Four viewports, binding four quarters each with a Pointset or a PointSubSet

    .. code-block:: py

        vis_obj = VisualizationVTK(number_of_viewports=4)
        vis_obj.SetRenderWindowName("Test - Four Viewport. PointSet and PointSubSet. Uniform Color.")
        vis_obj.BindTopLeft(pointset1, 'color', color=(255, 0, 0), new_layer=False)
        vis_obj.BindTopRight(pointset2, 'color', color=(0, 255, 0), new_layer=False)
        vis_obj.BindBottomLeft(pointsubset1, 'color', color=(255, 0, 0), new_layer=False)
        vis_obj.BindBottomRight(pointsubset2, 'color', color=(0, 255, 0), new_layer=False)

.. autoclass:: VisualizationVTK.VisualizationVTK
   :members:
   :undoc-members:

    .. rubric:: Methods

    .. autoautosummary:: VisualizationVTK.VisualizationVTK
        :methods:
