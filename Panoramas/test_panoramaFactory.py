from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from IOFactory import IOFactory
from PanoramaFactory import PanoramaFactory as pf


class TestPanoramaFactory(TestCase):
    def test_CreatePanorama_From_pts(self):
        try:
            pts = IOFactory.ReadPts(r'D:\Documents\Python Scripts\infragit\IO modules\test_pts2.pts')
            pts_pano = pf.CreatePanorama_byPoints(pts, elevationSpacing = 0.111, azimuthSpacing = 0.115, voidData = 180)
            plt.imshow(pts_pano.PanoramaImage, cmap = 'gray')
            plt.show()
        except:
            self.fail('Could not create panorama from points')

    def test_CreatePanorama_From_pts_intensity(self):
        try:
            ptslist = []
            colorlist = []
            pts = IOFactory.ReadPtx(r'D:\Documents\Python Scripts\infragit\IO modules\test_ptx2.ptx')
            pts_pano = pf.CreatePanorama_byPoints(pts[0], elevationSpacing = 0.111, azimuthSpacing = 0.115,
                                                  voidData = 0,
                                                  intensity = True)
            plt.figure()
            plt.imshow(pts_pano.PanoramaImage, vmax = 1., vmin = 0.)
            plt.show()
        except:
            self.fail('Could not create panorama with intensity')

    def test_CreatePanorama_From_property(self):
        try:

            colorlist = []
            pts = []
            IOFactory.ReadPts(r'D:\Documents\Python Scripts\infragit\Panoramas\panorama_test.pts', pts, colorlist,
                              merge = False)
            pts_pano = pf.CreatePanorama_byProperty(colorlist[0], elevationSpacing = 0.111, azimuthSpacing = 0.115,
                                                    voidData = 50)
        except:
            self.fail('Could not create panorama from property')
        plt.imshow(np.uint8(pts_pano.PanoramaImage))
        plt.show()
