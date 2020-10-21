from unittest import TestCase

from matplotlib import pyplot as plt

from IOFactory import IOFactory
from rasterTools import RasterTools


class TestRasterTools(TestCase):
    def test_slope_richdem(self):
        filename = r'../test_data/gully7e_11.txt'
        raster = IOFactory.rasterFromAscFile(filename)
        RasterTools.slope_richdem(raster, True, method='slope_radians')
        RasterTools.curvature_richdem(raster, True)
        RasterTools.plane_curvature_richdem(raster, True)
        RasterTools.profile_curvature_richdem(raster, True)
        plt.show()
