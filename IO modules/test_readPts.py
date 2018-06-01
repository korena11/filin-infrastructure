from unittest import TestCase

import ReadFunctions


class TestReadPts(TestCase):
    def test_ReadPts(self):
        pointsetlist = []
        colorslist = []
        points, colors = ReadFunctions.ReadPts(r'test_pts.pts', pointsetlist, colorslist, merge = False)
        print('passed seperate PointSets')
        # self.fail()

    def test_ReadPts_default(self):
        points, colors = ReadFunctions.ReadPts(r'test_pts.pts')
        print('Passed default - merging when some with color and some aren''t')
