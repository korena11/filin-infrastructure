from unittest import TestCase

import ReadFunctions


class TestReadPtx(TestCase):
    def test_ReadPtx(self):
        pointsetlist = []
        colorslist = []
        points, colors = ReadFunctions.ReadPtx(r'test_ptx.ptx', pointsetlist)
        print('passed seperate PointSets')
        # self.fail()

    # def test_ReadPts_default(self):
    #     points, colors = ReadFunctions.ReadPtx(r'test_ptx.ptx')
    #     print('Passed default - merging when some with color and some aren''t')
    #
    #     self.fail()
