from unittest import TestCase

from IOFactory import IOFactory
from PanoramaFactory import PanoramaFactory as pf


class TestPanoramaFactory(TestCase):
    def test_CreatePanorama_From_pts(self):
        pts = IOFactory.ReadPts(r'D:\Documents\Python Scripts\infragit\IO modules\test_pts.pts')
        pts_pano = pf.CreatePanorama(pts, elevationSpacing = 0.111, azimuthSpacing = 0.115)

        self.fail()
