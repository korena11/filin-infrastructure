from unittest import TestCase

import ReadFunctions
from ColorProperty import ColorProperty
from PointSet import PointSet


class TestReadPts(TestCase):
    def test_ReadPts(self):
        pointsetlist = []
        colorslist = []
        filename = r'test_readPts.pts'
        points = ReadFunctions.ReadPts(filename, pointsetlist, colorslist, merge = False)

        self.assertEqual(len(points), 3, 'Not all point clouds were loaded in different clouds')
        self.assertIsInstance(pointsetlist[0], PointSet, 'Point set was not loaded into PointSet')
        self.assertIsNotNone(pointsetlist[0].Intensity, 'Intensity was not loaded')
        if len(colorslist) > 0:
            self.assertIsInstance(colorslist[0], ColorProperty, 'Color was not loaded into ColorProperty ')
        self.assertEqual(pointsetlist[0].Size, 782, 'not all point cloud was read')
        self.assertEqual(pointsetlist[2].Size, 31591, 'last point cloud was not read properly')
        self.assertEqual(filename, pointsetlist[0].path, 'path was not set')

        print('passed with lists')
        print('passed seperate PointSets')
        # self.fail()

    def test_ReadPts_default(self):
        filename = r'test_readPts.pts'
        points = ReadFunctions.ReadPts(filename)
        self.assertIsInstance(points, PointSet, 'Point set was not loaded into PointSet')
        self.assertIsNotNone(points.Intensity, 'Intensity was not loaded')
        self.assertEqual(points.Size, 502616, 'not all point cloud was read')

        self.assertEqual(filename, points.path, 'path was not set')

        print('Passed default - merging when some with color and some aren''t')
