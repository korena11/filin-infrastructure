import cProfile
import io
import pstats
from unittest import TestCase

from IOFactory import IOFactory
from NeighborsFactory import NeighborsFactory


class TestNeighborsFactory(TestCase):
    # def test_pointSetOpen3D_byRadius_kdTree(self):
    #     from PointSetOpen3D import PointSetOpen3D
    #     pr = cProfile.Profile()
    #     pr.enable()
    #     colors = []
    #     pts = []
    #
    #     # for neighbors and normal computations
    #     folderPath = '../../test_data/'
    #     dataName = 'test_pts'
    #
    #     search_radius = 0.25
    #     pcl = IOFactory.ReadPts(folderPath + dataName + '.pts', pts, colors, merge=False)
    #     pcd = PointSetOpen3D(pcl[0])
    #     neighbors = NeighborsFactory.pointSetOpen3D_radius_kdTree(pcd, search_radius)
    #
    #     neighbors_restricted = NeighborsFactory.pointSetOpen3D_radius_kdTree(pcd, search_radius, max_neighbors=20)

    def test_pointSetOpen3D_knn_kdTree(self):
        from PointSetOpen3D import PointSetOpen3D
        pr = cProfile.Profile()
        pr.enable()
        colors = []
        pts = []

        # for neighbors and normal computations
        # folderPath = '../../test_data/'
        # dataName = 'test_pts'
        folderPath = '/home/reuma/ownCloud/Data/ISPRSJ_paper/'
        dataName = 'gesher_wall'
        knn = 15
        pcl = IOFactory.ReadPts(folderPath + dataName + '.pts', pts, colors, merge=False)
        pcd = PointSetOpen3D(pcl[0])
        # neighbors_e = NeighborsFactory.CalculateAllPointsNeighbors(pcd, -1, 15)
        neighbors = NeighborsFactory.pointSetOpen3D_knn_kdTree(pcd, knn)

        # neighbors_restricted = NeighborsFactory.pointSetOpen3D_rknn_kdTree(pcd, knn, max_radius=0.05)

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.print_stats()
        print(s.getvalue())

    def test_balltreePointSet_rnn(self):
        from BallTreePointSet import BallTreePointSet
        pr = cProfile.Profile()
        pr.enable()
        colors = []
        pts = []

        # for neighbors and normal computations
        folderPath = '../../test_data/'
        dataName = 'test_pts'

        search_radius = 0.25
        pcl = IOFactory.ReadPts(folderPath + dataName + '.pts', pts, colors, merge=False)
        pcd = BallTreePointSet(pcl[0], leaf_size=10)
        neighbors = NeighborsFactory.balltreePointSet_rnn(pcd, search_radius)

        print('hello')

    def test_buildNeighbors_rnn(self):
        from PointSetOpen3D import PointSetOpen3D
        pr = cProfile.Profile()
        pr.enable()
        colors = []
        pts = []

        # for neighbors and normal computations
        folderPath = '../../test_data/'
        dataName = 'test_pts'

        search_radius = 0.25
        pcl = IOFactory.ReadPts(folderPath + dataName + '.pts', pts, colors, merge=False)
        pcd = PointSetOpen3D(pcl[0])
        neighbors = NeighborsFactory.buildNeighbors_rnn(pcl[0], search_radius,
                                                        method=NeighborsFactory.pointSetOpen3D_rnn_kdTree)

        print('hello')

    def test_buildNeighbors_knn(self):
        from PointSetOpen3D import PointSetOpen3D
        pr = cProfile.Profile()
        pr.enable()
        colors = []
        pts = []

        # for neighbors and normal computations
        folderPath = '../../test_data/'
        dataName = 'test_pts'

        knn = 50
        pcl = IOFactory.ReadPts(folderPath + dataName + '.pts', pts, colors, merge=False)
        pcd = PointSetOpen3D(pcl[0])
        neighbors = NeighborsFactory.buildNeighbors_knn(pcl[0], knn, method=NeighborsFactory.pointSetOpen3D_knn_kdTree)
