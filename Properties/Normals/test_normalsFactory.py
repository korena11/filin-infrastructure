from unittest import TestCase

import matplotlib.pyplot as plt

from IOFactory import IOFactory
from NormalsFactory import NormalsFactory


class TestNormalsFactory(TestCase):
    # def test_normals_from_panorama(self):
    #     filename = r'../IOmodules/test_pts.pts'
    #     points = IOFactory.ReadPts(filename)
    #     panorama = PanoramaFactory.CreatePanorama_byPoints(points, elevationSpacing = 0.111, azimuthSpacing = 0.115,
    #                                                        voidData = 30)
    #
    #     NormalsFactory.normalsComputation_in_raster(panorama[:,0], panorama[:,1], panorama[:,2])

    def test_normalsPCA_no_tree_radius_kneighbors(self):
        filename = r'../IOmodules/test_pts.pts'
        points = IOFactory.ReadPts(filename)
        normals = NormalsFactory.normalsPCA(points, radius=0.5)

        normals2 = NormalsFactory.normalsPCA(points, k_neighbors=5)

        normals.save('normals.h5')
        normals2.save('normals2.h5')
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        ax.scatter(points.X, points.Y, zs=points.Z)
        ax2.scatter(points.X, points.Y, zs=points.Z)

        ax.quiver(points.X, points.Y, points.Z, normals.dX, normals.dY, normals.dZ)
        ax2.quiver(points.X, points.Y, points.Z, normals2.dX, normals2.dY, normals2.dZ)

        plt.show()
