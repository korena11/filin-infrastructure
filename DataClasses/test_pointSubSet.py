from unittest import TestCase

import numpy as np

from IOFactory import IOFactory
from PointSubSet import PointSubSet


class TestPointSubSet(TestCase):

    def test_Size(self):
        pcl = IOFactory.ReadPts('../test_data/test_pts2.pts', merge=False)
        indices = np.arange(0, pcl[0].Size, 10)
        subset = PointSubSet(pcl[0], indices)
        self.assertEqual(subset.Size, 885, 'Size of subset incompatible')
        numpy_ = isinstance(subset.ToNumpy(), np.ndarray)

        self.assertTrue(numpy_)
        self.assertEqual(subset.Intensity.shape[0], 885, 'Intensity not returned correctly')
        print(subset.Intensity)
