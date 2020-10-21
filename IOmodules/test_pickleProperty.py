from unittest import TestCase

import numpy as np

from CurvatureFactory import CurvatureFactory, CurvatureProperty
from IOFactory import IOFactory
from NeighborsFactory import NeighborsFactory
from NormalsProperty import NormalsProperty
from PointSetOpen3D import PointSetOpen3D


class TestPickleProperty(TestCase):

    def test_pickleProperty(self):
        folder = '../test_data/'
        filename = 'test_pts.pts'
        color = []
        pts = IOFactory.ReadPts(folder + filename, colorslist=color, merge=False)
        color_property = color[0]
        pts = pts[0]
        p3d = PointSetOpen3D(pts)
        p3d.CalculateNormals()

        normals = NormalsProperty(pts, np.asarray(p3d.data.normals))
        neighbors = NeighborsFactory.pointSetOpen3D_knn_kdTree(p3d, 50)
        curvatureProperty = CurvatureFactory.pointSetOpen3D_3parameters(p3d, neighbors)
        CurvatureFactory.umbrella_curvature(neighbors, normals, curvatureProperty=curvatureProperty)
        neighbors._BaseProperty__dataset = pts
        # SaveFunctions.pickleProperty(color_property, 'test_pts.p', save_dataset=True)
        IOFactory.saveDataset(pts, 'test_pts.p')

        dataset = IOFactory.load('test_pts_curvature.p', CurvatureProperty)
        print('hello')
