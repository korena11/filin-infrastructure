from unittest import TestCase

from CurvatureFactory import CurvatureFactory
from IOFactory import IOFactory
from NeighborsFactory import NeighborsFactory
from PointSetOpen3D import PointSetOpen3D
from VisualizationO3D import VisualizationO3D


class TestVisualizationO3D(TestCase):

    def test_visualize_pointset(self):
        # create an empty list for the colors to be loaded into
        colors = []
        # load the point cloud for visualization
        pcl = IOFactory.ReadPts('../test_data/test_pts2.pts', colorslist=colors, merge=False)

        # visualize the PointSet with a color property
        VisualizationO3D.visualize_pointset(pcl[0], colors[0])

    def test_visualize_property(self):
        colors = []
        # load the point cloud for visualization
        pcl = IOFactory.ReadPts('../test_data/test_pts2.pts', colorslist=colors, merge=False)
        pcd = PointSetOpen3D(pcl[0])  # convert to PointSetOpen3D

        # Prepare the property to show
        neighbors = NeighborsFactory.CalculateAllPointsNeighbors(pcd, .1, -1)
        curvature = CurvatureFactory.pointSetOpen3D_3parameters(pcd, neighbors, valid_sectors=4)

        # Create a VisualizationO3D object
        visualization = VisualizationO3D()
        visualization.visualize_property(curvature)  # send property for visualization
