from unittest import TestCase

from IOFactory import IOFactory
from PanoramaFactory import PanoramaFactory


class TestPanoramaProperty(TestCase):
    # def test_make_rectangular(self):

    # pts = IOFactory.ReadPts(r'..\IO modules\test_pts.pts')
    #
    # pts_panorama = PanoramaFactory.CreatePanorama_byPoints(pts, elevationSpacing = 0.111, azimuthSpacing = 0.115,
    #                                             voidData = 30)
    # img_orig = pts_panorama.PanoramaImage
    # plt.title('panorama')
    # plt.imshow(img_orig)
    # plt.show()
    #
    # subset = pts_panorama.extract_area((7, 37), (27,57))
    # subset_panorama = PanoramaFactory.CreatePanorama_byPoints(subset,
    #                                                           elevationSpacing = pts_panorama.elevation_spacing,
    #                                                           azimuthSpacing = pts_panorama.azimuth_spacing,
    #                                                           voidData = pts_panorama.void_data)
    # img_orig = subset_panorama.PanoramaImage
    # plt.title('subset panorama')
    # plt.imshow(img_orig)
    # plt.show()

    def test_indexes_to_panorama(self):
        print('hello')
        pts = IOFactory.ReadPts(r'..\IO modules\test_pts.pts')

        pts_panorama = PanoramaFactory.CreatePanorama_byPoints(pts, elevationSpacing = 0.111, azimuthSpacing = 0.115,
                                                               voidData = 30)
        pts_panorama.indexes_to_panorama()

        self.assertIsNotNone(pts_panorama.Points.GetPoint(int(pts_panorama.PanoramaImage[0, 0])))
