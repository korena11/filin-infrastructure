from unittest import TestCase
import Utils.GeometryUtils as gt
import Utils.MyTools as mt
from IOmodules.IOFactory import IOFactory


class TestCurve2shapely(TestCase):
    def test_curve2shapely(self):
        import matplotlib.pyplot as plt
        from shapely_polygon.geometry import asMultiPoint
        pts = IOFactory.ReadPts('../test_data/bulbus - 2k6.pts')
        ps = asMultiPoint(pts.ToNumpy()[:, :2])
        # x = [1, 2, 3, 4]
        # y = [1, 2, 3, 4]
        # m = [[15, 14, 13, 12], [14, 12, 10, 8], [13, 10, 7, 4], [12, 8, 4, 0]]
        z = mt.interpolate_data(pts.ToNumpy(), pts.Z, .01)
        cs = plt.contour(z)
        curves = gt.curve2shapely(cs, (pts.X.min(), pts.Y.min()))
        for curve in curves:
            print(curve.contains)
        circ = gt.circularity_measure(curves[1])


        # self.fail()
