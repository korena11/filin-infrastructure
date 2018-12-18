import cProfile
import io
import pstats
from unittest import TestCase

from CurvatureFactory import CurvatureFactory
from IOFactory import IOFactory
from NeighborsFactory import NeighborsFactory
from PanoramaFactory import PanoramaFactory
from PointSetOpen3D import PointSetOpen3D


class TestCurvatureFactory(TestCase):
    def test_read_or_calculate_curvature_data(self):
        pr = cProfile.Profile()
        pr.enable()
        colors = []
        pts = []

        # for curvature and normal computations
        folderPath = '../../test_data/'
        dataName = 'test_pts'

        search_radius = 0.25
        max_nn = -1
        localNeighborhoodParams = {'search_radius': search_radius, 'maxNN': max_nn}
        # curvatureFilePath = IOFactory.GetCurvatureFilePath(folderPath, dataName, 3,
        #                                                    localNeighborhoodParameters=localNeighborhoodParams,
        #                                                    decimationRadius=0, testRun=False)

        pcl = IOFactory.ReadPts(folderPath + dataName + '.pts',
                                pts, colors, merge=False)
        p3d = PointSetOpen3D(pcl[0])
        # p3d.CalculateNormals(localNeighborhoodParams['search_radius'], localNeighborhoodParams['maxNN'])
        neighborsProperty = NeighborsFactory.CalculateAllPointsNeighbors(p3d, **localNeighborhoodParams)
        curvature = CurvatureFactory.curvature_PointSetOpen3D(p3d, neighborsProperty, valid_sectors=4)
        print('hello')
        curvature_panorama = PanoramaFactory.CreatePanorama_byProperty(curvature, elevationSpacing=0.111,
                                                                       azimuthSpacing=0.115, voidData=30)
        import matplotlib.pyplot as plt
        plt.imshow(curvature_panorama.PanoramaImage.astype('f'))
        plt.show()
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.print_stats()
        print(s.getvalue())

    def test_curvature_raster_fundamentalForm(self):
        # for curvature and normal computations
        folderPath = '/home/reuma/ownCloud/Data/RS_paper/'
        dataName = 'gully7_05'

        raster = IOFactory.rasterFromAscFile(folderPath + dataName + '.txt')
        curvature = CurvatureFactory.curvature_raster_fundamentalForm(raster, ksize=3, gradientType='L2',
                                                                      sigma=9)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1)
        ax[0].imshow(curvature.k1)
        ax[1].imshow(curvature.k2)
        ax[2].imshow(curvature.mean_curvature)
        ax[3].imshow(curvature.curvadness)

        plt.show()
