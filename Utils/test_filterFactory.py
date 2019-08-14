import cProfile
from unittest import TestCase

from FilterFactory import FilterFactory
from IOmodules.IOFactory import IOFactory
from VisualizationClasses.VisualizationO3D import VisualizationO3D


class TestFilterFactory(TestCase):
    # def test_SmoothPointSet_MLS(self):
    #     pr = cProfile.Profile()
    #     pr.enable()
    #     colors = []
    #     pts = []
    #     # for curvature and normal computations
    #     folderPath = '../test_data/'
    #     dataName = 'test_pts'
    #     vis = VisualizationO3D()
    #     pcl = IOFactory.ReadPts(folderPath + dataName + '.pts',
    #                             pts, colors, merge=False)
    #
    #     pcl_smoothed = FilterFactory.SmoothPointSet_MLS(pcl, 0.1, 2, False)
    #     vis.visualize_pointset(pcl_smoothed)
    #     self.fail()

    def test_smooth_simple(self):
        from Properties.Neighborhood.NeighborsFactory import NeighborsFactory
        from DataClasses.KdTreePointSet import KdTreePointSet
        import numpy as np
        pr = cProfile.Profile()
        pr.enable()
        colors = []
        pts = []
        # for curvature and normal computations
        folderPath = '../test_data/'
        dataName = 'bunny_20 - Cloud'
        vis = VisualizationO3D()
        pcl = IOFactory.ReadPts(folderPath + dataName + '.pts',
                                pts, colors, merge=True)
        kdt_pcl = KdTreePointSet(pcl)
        neighborhood = NeighborsFactory.kdtreePointSet_rnn(kdt_pcl, 0.5)
        pcl_smoothed = FilterFactory.smooth_simple(neighborhood)
        np.savetxt(folderPath + dataName + 'smoothed.txt', pcl_smoothed.ToNumpy())
        vis.visualize_pointset(pcl_smoothed)
        # self.fail()
