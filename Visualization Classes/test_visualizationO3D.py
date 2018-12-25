from unittest import TestCase

from CurvatureFactory import CurvatureFactory
from IOFactory import IOFactory
from NeighborsFactory import NeighborsFactory
from PointSetOpen3D import PointSetOpen3D
from VisualizationO3D import VisualizationO3D


class TestVisualizationO3D(TestCase):

    # def test_visualize_pointset(self):
    #     colors = []
    #     pcl = IOFactory.ReadPts('../test_data/test_pts2.pts',colorslist=colors, merge=False)
    #     VisualizationO3D.visualize_pointset(pcl[0], colors[0])

    def test_visualize_property(self):
        colors = []
        pcl = IOFactory.ReadPts('../test_data/test_pts2.pts', colorslist=colors, merge=False)

        pcd = PointSetOpen3D(pcl[0])
        neighbors = NeighborsFactory.CalculateAllPointsNeighbors(pcd, .1, -1)
        curvature = CurvatureFactory.curvature_PointSetOpen3D(pcd, neighbors, valid_sectors=4)
        visualization = VisualizationO3D()
        visualization.visualize_property(curvature)
