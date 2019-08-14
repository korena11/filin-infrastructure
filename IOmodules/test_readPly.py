from unittest import TestCase

import ReadFunctions
from PointSet import PointSet


class TestReadPly(TestCase):
    def test_ReadPly_default(self):
        filename = '../test_data/test_ply.ply'
        data = ReadFunctions.ReadPly(filename)

        self.assertIsInstance(data, tuple, 'Returned object is not a tuple')
        self.assertIsInstance(data[0], PointSet, 'First returned object is not a \'PointSet\' object')
        self.assertIsInstance(data[1], dict, 'Second returned object is not a dictionary')
        self.assertEqual(len(data), 2, 'Unexpected number of objects returned from method')

        self.assertEqual(filename, data[0].path, 'path was not set')
        self.assertEqual(data[0].Size, 371973, 'Unexpected number of points in the loaded file')

        print('Passed reading *.ply file test')
