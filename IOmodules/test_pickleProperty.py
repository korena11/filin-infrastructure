from unittest import TestCase

from IOFactory import IOFactory
from PointSet import PointSet


class TestPickleProperty(TestCase):

    def test_pickleProperty(self):
        folder = '../test_data/'
        filename = 'test_pts.pts'
        color = []
        pts = IOFactory.ReadPts(folder + filename, colorslist=color, merge=False)
        color_property = color[0]
        pts = pts[0]
        # SaveFunctions.pickleProperty(color_property, 'test_pts.p', save_dataset=True)
        color_property.save('test_pts.p', save_dataset=True)

        dataset = IOFactory.load('test_pts_data.p', PointSet)
