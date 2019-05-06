from unittest import TestCase

from Visualization import VisualizationVTK

from CurvatureFactory import CurvatureFactory
from IOFactory import IOFactory
from NeighborsFactory import NeighborsFactory
from PointSetOpen3D import PointSetOpen3D


class TestVisualizationVTK(TestCase):
    def test_Bind(self):
        pts = []
        colors = []
        # for curvature and normal computations
        folderPath = '../test_data/'
        dataName = 'test_pts'

        search_radius = 0.25
        max_nn = -1
        localNeighborhoodParams = {'search_radius': search_radius, 'maxNN': max_nn}

        pcl = IOFactory.ReadPts(folderPath + dataName + '.pts',
                                pts, colors, merge=False)
        p3d = PointSetOpen3D(pcl[0])
        neighborsProperty = NeighborsFactory.CalculateAllPointsNeighbors(p3d, **localNeighborhoodParams)
        curvature = CurvatureFactory.pointSetOpen3D_3parameters(p3d, neighborsProperty, min_points_in_neighborhood=2,
                                                                valid_sectors=4,
                                                                verbose=False)

        vis_obj = VisualizationVTK()
        import numpy as np
        # vis_obj.SetRenderWindowBackgroundColor((0,0,0))
        vis_obj.Bind(input_data=curvature, render_flag='parametricColor', color=(np.abs(curvature.k1)),
                     new_layer=False)
        vis_obj.Show()
        print('hello')

    def TestPointSize(self):
        import numpy as np
        from PointSet import PointSet
        from PointSubSet import PointSubSet
        points1 = (np.random.rand(1000, 3) - 0.5) * 1000.0
        pointset1 = PointSet(points=points1)
        pointsubset1 = PointSubSet(points=pointset1, indices=list(range(0, len(points1), 3)))

        points2 = (np.random.rand(1000, 3) - 0.5) * 1000.0
        pointset2 = PointSet(points=points2)
        pointsubset2 = PointSubSet(points=pointset2, indices=list(range(0, len(points2), 3)))

        visualization_object = VisualizationVTK(number_of_view_ports=2, two_viewports_vertical_horizontal='V')
        visualization_object.SetRenderWindowName("Testing Point Size")

        visualization_object.BindFirstHalf(input_data=pointset1, render_flag='color', color=(255, 0, 0), point_size=5,
                                           new_layer=False)
        visualization_object.BindFirstHalf(input_data=pointset2, render_flag='color', color=(0, 255, 0), point_size=5,
                                           new_layer=True)

        visualization_object.BindSecondHalf(input_data=pointsubset1, render_flag='color', color=(255, 0, 0),
                                            point_size=5,
                                            new_layer=False)
        visualization_object.BindSecondHalf(input_data=pointsubset2, render_flag='color', color=(0, 255, 0),
                                            point_size=5,
                                            new_layer=True)

        visualization_object.Show()

    def TestBackgroundColors(self):
        visualization_object = VisualizationVTK(number_of_view_ports=4)
        visualization_object.SetRenderWindowName("Red Background Color for All.")
        background_color = (255, 0, 0)
        visualization_object.SetRenderWindowBackgroundColor(background_color)
        visualization_object.Show()

        visualization_object.SetRenderWindowName("TopLeft is Red, BottomRight is Blue, other two are default (black).")
        background_color = [(255, 0, 0), None, None, (0, 0, 255)]
        visualization_object.SetRenderWindowBackgroundColor(background_color)
        visualization_object.Show()

        visualization_object.SetRenderWindowName("Each quarter different color.")
        background_color = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]
        visualization_object.SetRenderWindowBackgroundColor(background_color)
        visualization_object.Show()
