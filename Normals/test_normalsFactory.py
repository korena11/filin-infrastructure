from unittest import TestCase

from IOFactory import IOFactory
from NormalsFactory import NormalsFactory
from PanoramaFactory import PanoramaFactory


class TestNormalsFactory(TestCase):
    def test_normals_from_panorama(self):
        filename = r'../IO modules/test_pts.pts'
        points = IOFactory.ReadPts(filename)
        panorama = PanoramaFactory.CreatePanorama_byPoints(points, elevationSpacing = 0.111, azimuthSpacing = 0.115,
                                                           voidData = 30)

        NormalsFactory.normals_from_panorama(panorama)
