from unittest import TestCase

import ReadFunctions
from ColorProperty import ColorProperty
from IOFactory import IOFactory
from PointSet import PointSet
from Transformations.TransformationMatrixProperty import TransformationMatrixProperty


class TestReadPtx(TestCase):
    def test_ReadPtx_lists(self):
        pointsetlist = []
        colorslist = []
        transMatrices = []
        filename = r'test_ptx2.ptx'
        ReadFunctions.ReadPtx(filename, pointsetlist=pointsetlist, colorslist=colorslist,
                              trasformationMatrices=transMatrices)

        self.assertIsInstance(pointsetlist[0], PointSet, 'Point set was not loaded into PointSet')
        self.assertIsNotNone(pointsetlist[0].Intensity, 'Intensity was not loaded')
        if len(colorslist) > 0:
            self.assertIsInstance(colorslist[0], ColorProperty, 'Color was not loaded into ColorProperty ')
        if len(transMatrices) > 0:
            self.assertIsInstance(transMatrices[0], TransformationMatrixProperty, 'Transformation Matrix was not'
                                                                                  'loaded into '
                                                                                  'TransformationMatrixProperty')
        self.assertEqual(filename, pointsetlist[0].path, 'path was not set')

        print('passed with lists')

        # self.fail()

    def test_factory_ReadPtx_lists(self):
        pointsetlist = []
        colorslist = []
        transMatrices = []
        IOFactory.ReadPtx(r'test_ptx2.ptx', pointsetlist=pointsetlist, colorslist=colorslist,
                          trasformationMatrices=transMatrices)

        self.assertIsInstance(pointsetlist[0], PointSet, 'Point set was not loaded into PointSet')
        self.assertIsNotNone(pointsetlist[0].Intensity, 'Intensity was not loaded')
        if len(colorslist) > 0:
            self.assertIsInstance(colorslist[0], ColorProperty, 'Color was not loaded into ColorProperty ')
        if len(transMatrices) > 0:
            self.assertIsInstance(transMatrices[0], TransformationMatrixProperty, 'Transformation Matrix was not'
                                                                                  'loaded into '
                                                                                  'TransformationMatrixProperty')

        print('passed with lists through IOFactory')

    def test_ReadPtx_vs_ReadPts(self):
        pointsetlist_ptx = []
        colorslist_ptx = []
        transMatrices_ptx = []

        pointsetlist_pts = []
        colorslist_pts = []

        IOFactory.ReadPtx(r'test_ptx2.ptx', pointsetlist=pointsetlist_ptx, colorslist=colorslist_ptx,
                          trasformationMatrices=transMatrices_ptx)

        IOFactory.ReadPts(r'test_pts2.pts', pointsetlist_pts, colorslist_pts, merge = False)

        self.assertEqual(len(pointsetlist_pts), len(pointsetlist_ptx), 'pts did not separate all point clouds')
        self.assertEqual(pointsetlist_pts[0].Size, pointsetlist_ptx[0].Size, 'ptx did not load'
                                                                             'all points')
        self.assertEqual(len(colorslist_pts), len(colorslist_ptx))

        print('Read ptx passed')
