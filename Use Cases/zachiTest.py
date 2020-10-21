"""
Code by Zachi for testing various stuff, DO NOT USE AS USE CASE!
"""

from logging import info, basicConfig, DEBUG

import numpy as np
from numpy import meshgrid, arange

from KdTreePointSet import KdTreePointSet
from NeighborsFactory import NeighborsFactory
from NormalsFactory import NormalsFactory
from PointSetOpen3D import PointSetOpen3D
from VisualizationO3D import VisualizationO3D


def z(x, y, a3, a4, a5):
    return a3 * (x ** 2) + a4 * x * y + a5 * (y ** 2) + 5


def retPoints(n, lx=5, ly=5, random=True):
    a3 = 0.1
    a4 = 0.02
    a5 = 0.3
    if random:
        x = lx * np.random.randn(n)
        y = ly * np.random.randn(n)
    else:
        x = arange(-lx, lx, 0.5)
        y = arange(-ly, ly, 0.5)
        xs, ys = meshgrid(x, y)
        x = xs.reshape((-1,))
        y = ys.reshape((-1,))
    points = np.vstack([x, y, z(x, y, a3, a4, a5)]).T

    return points


if __name__ == '__main__':
    basicConfig(format='%(asctime)s %(message)s', level=DEBUG)

    xyz = retPoints(1000, random=False)
    xyz = PointSetOpen3D(xyz)

    normalProp = NormalsFactory.normals_open3D(xyz, search_radius=2)

    # # filename = 'F:\\Dropbox\\Research\\Code\\Segmentation\\data\\agriculture1_clean.pts'
    # # xyz = IOFactory.ReadPts(filename)
    # # print(xyz.Size)
    # # print(type(xyz))
    #
    # filename = 'F:\\Dropbox\\Research\\Code\\Segmentation\\data\\rueMadame_database\\GT_Madame1_3.ply'
    # filename = '../test_data/test_ply.ply'
    # info('Reading ply file')
    # # plyData = PlyData.read(filename)  # Reading ply file
    # # properties = list(map(lambda p: p.name, plyData['vertex'].properties))  # Getting list of properties of vertices
    # # data = plyData['vertex'].data
    #
    # # Extracting the 3-D coordinates of the points
    # xyz = IOFactory.ReadPly(filename, False)
    #
    # # info('Creating PointSetOpen3D')
    # # temp = PointSetOpen3D(xyz)
    # #
    # # info('neighbor querying for a single point')
    # # temp.kdTreeOpen3D.search_radius_vector_3d(temp.ToNumpy()[0], 1)
    # # info('neighbor querying for entire point set')
    # # tmp = list(map(lambda pnt: temp.kdTreeOpen3D.search_knn_vector_3d(pnt, 10), temp.ToNumpy()))
    #
    info('Creating KdTreePointSet')
    kdTreePntSet = KdTreePointSet(xyz, leaf_size=10)

    info('neighbor querying for entire point set')
    kdTreePntSet.query(kdTreePntSet.ToNumpy(), 10)
    info('querying complete')

    info('Creating neighbor property for the point set')
    temp = NeighborsFactory.kdtreePointSet_knn(kdTreePntSet, 16)

    info('Creating tensor property')
    from TensorFactory import TensorFactory

    tensors = TensorFactory.computeTensorsProperty_givenNeighborhood(xyz, temp)

    info('Extracting normals from tensors')
    normals = NormalsFactory.normals_from_tensors(tensors)
    # normalTest = array(list(map(dot, normals.Normals, normalProp.Normals)))
    # print(normalTest)
    # print(nonzero(abs(normalTest) < 1))

    # visObj = VisualizationO3D()
    # visObj.visualize_property(normals)
    #
    # visObj = VisualizationO3D()
    # visObj.visualize_property(normalProp)

    info('Computing umbrella curvature')
    from CurvatureFactory import CurvatureFactory

    uc = CurvatureFactory.umbrella_curvature(temp, normals)
    ucVals = uc.umbrella_curvature
    # print(unique(ucVals))

    import open3d as o3d
    from CurvatureProperty import CurvatureProperty

    uc = CurvatureProperty(PointSetOpen3D(uc.Points), umbrella_curvature=uc.umbrella_curvature)
    uc.Points.data.normals = o3d.Vector3dVector(normals.Normals)

    visObj = VisualizationO3D()
    visObj.visualize_property(uc)
