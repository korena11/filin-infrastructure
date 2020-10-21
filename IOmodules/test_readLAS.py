from unittest import TestCase

from PointSetOpen3D import PointSetOpen3D
from ReadFunctions import ReadLAS


class TestReadLAS(TestCase):
    def test_ReadLAS(self):
        filename = '../test_data/minnesota.laz'
        pcl = ReadLAS(filename)
        pclo3d = PointSetOpen3D(pcl)
        pclo3d.Visualize()
