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
        folderPath = '../../test_data/'
        dataName = 'test_pts'

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



    def test_buildNeighbors_panorama(self):
        from Properties.Panoramas.PanoramaFactory import PanoramaFactory
        from Properties.Curvature.CurvatureFactory import CurvatureFactory
        from VisualizationO3D import VisualizationO3D
        from DataClasses.PointSetOpen3D import PointSetOpen3D
        import numpy as np
        from Properties.Normals.NormalsFactory import NormalsFactory, NormalsProperty
        pr = cProfile.Profile()
        pr.enable()
        colors = []
        pts = []

        # for neighbors and normal computations
        folderPath = '../../test_data/'
        dataName = 'bulbus'

        pcl= IOFactory.ReadPts(folderPath + dataName + '.pts', pts, colors, merge=False)
        o3d = PointSetOpen3D(pcl[0])
        o3d.CalculateNormals(0.03)

        normals_tmp = np.asarray(o3d.data.normals)
        # normals_tmp[normals_tmp[:,2]<0] = -normals_tmp[normals_tmp[:,2]<0]
        normals = NormalsProperty(pcl[0], normals_tmp)
        panorama = PanoramaFactory.CreatePanorama(colors[0], azimuthSpacing=0.02, elevationSpacing=0.02)

        neighbors = NeighborsFactory.buildNeighbors_panorama(panorama, .5)

        # for neighbor in neighbors:
        #     if neighbor is not None:
                # print('index {}, distances {}'.format(neighbor.center_point_idx, neighbor.distances))

        curvature = CurvatureFactory.umbrella_curvature(neighbors, normals, min_points_in_neighborhood=0, min_points_in_sector=0, valid_sectors=4, num_sectors=8, invalid_value=0)
        vis = VisualizationO3D()
        vis.visualize_property(curvature)


