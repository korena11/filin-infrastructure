import cProfile
from unittest import TestCase

import MyTools as mt
from IOFactory import IOFactory


class TestBallTree_saliency(TestCase):
    def test_ballTree_saliency(self):
        pr = cProfile.Profile()
        pr.enable()
        colors = []
        pts = []
        # for curvature and normal computations
        folderPath = '../test_data/'
        dataName = 'test_pts'

        pcl = IOFactory.ReadPts(folderPath + dataName + '.pts',
                                pts, colors, merge=False)
        pcl_saliency, pcl_curvature, pcl_normals = mt.ballTree_saliency(pcl[0], .1, 10, 'mean_curvature')
        from VisualizationO3D import VisualizationO3D
        vis = VisualizationO3D()
        vis.visualize_property(pcl_saliency)
        print('hello')
