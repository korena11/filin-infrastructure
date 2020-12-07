from unittest import TestCase

from DataClasses.PointSetOpen3D import PointSetOpen3D
from IOmodules.IOFactory import IOFactory
from Properties.Curvature.CurvatureFactory import CurvatureFactory
from Properties.Neighborhood.NeighborsFactory import NeighborsFactory, NeighborsProperty
from VisualizationClasses.VisualizationO3D import VisualizationO3D


class TestVisualizationO3D(TestCase):

    # def test_visualize_pointset(self):
    #     # create an empty list for the colors to be loaded into
    #     colors = []
    #     # load the point cloud for visualization
    #     pcl = IOFactory.ReadPts('../test_data/test_pts2.pts', colorslist=colors, merge=False)
    #
    #     # visualize the PointSet with a color property
    #     VisualizationO3D.visualize_pointset(pcl[0], colors[0])
    #
    # def test_visualize_property(self):
    #     colors = []
    #     # load the point cloud for visualization
    #     pcl = IOFactory.ReadPts('../test_data/test_pts2.pts', colorslist=colors, merge=False)
    #     pcd = PointSetOpen3D(pcl[0])  # convert to PointSetOpen3D
    #
    #     # Prepare the property to show
    #     neighbors = NeighborsFactory.CalculateAllPointsNeighbors(pcd, .1, -1)
    #     curvature = CurvatureFactory.pointSetOpen3D_3parameters(pcd, neighbors, valid_sectors=4)
    #
    #     # Create a VisualizationO3D object
    #     visualization = VisualizationO3D()
    #     visualization.visualize_property(curvature)  # send property for visualization

    def test_visualize_neighborhood(self):
        import numpy as np
        import Properties.Neighborhood.WeightingFunctions as wf
        vis = VisualizationO3D()
        r = 3
        nn_max = 200
        noise = 0

        # 1. Build 3 planes
        x = np.arange(0, 75, 1)
        x_ = np.arange(0, 25, 1)
        xx, yy = np.meshgrid(x, x)
        xx_, yy_ = np.meshgrid(x_, x_)

        z1 = np.zeros(xx.shape)
        z2 = np.ones(xx_.shape) * 5
        z3 = np.ones(xx_.shape) * 10

        p1 = np.vstack((xx.flatten(), yy.flatten(), z1.flatten()))
        p2 = np.vstack((xx_.flatten(), yy_.flatten() + 25, z2.flatten()))
        p3 = np.vstack((xx_.flatten() + 50, yy_.flatten() + 30, z3.flatten()))

        planes = np.hstack((p3, p2, p1)).T

        # remove points on z=0 that share x,y with the two other planes
        p_, ind = np.unique(planes[:, :2], axis=0, return_index=True)
        pts = PointSetOpen3D(planes[ind, :])
        size = []
        radi = []
        # compute neighborhood, normals and curvature
        # for i in np.arange(1,20):
        # neighborhood = NeighborsFactory.pointSetOpen3D_rnn_kdTree(pts, i)
        #     size.append(neighborhood.average_neighborhood_size())
        #     radi.append(neighborhood.average_neighborhood_radius())
        # plt.plot(np.arange(1, 20), np.asarray(size))
        # plt.plot(np.arange(1, 20), np.asarray(radi))

        neighborhood = NeighborsFactory.ComputeNeighbors_raster(pts, 1, r)
        weighted_neighborhood = NeighborsProperty(pts)
        for neighbors in neighborhood:
            neighbors.weightNeighborhood(wf.laplacianWeights,  3.)
        vis.visualize_neighborhoods(neighborhood)