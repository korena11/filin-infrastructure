# import mayavi.mlab as mlab
import sys
from warnings import warn

import numpy as np
from numpy import dtype, genfromtxt
from sklearn.neighbors import BallTree, KDTree

from Cuda.cuda_API import *
from Utils import MyTools as mt
from DataClasses.PointSet import PointSet
from DataClasses.PointSetOpen3D import PointSetOpen3D
from Properties.Normals.NormalsProperty import NormalsProperty
from Properties.Neighborhood.NeighborsProperty import NeighborsProperty

if sys.platform == 'linux':
    pass


class NormalsFactory:

    @classmethod
    def normals_from_tensors(cls, tensorProperty):
        """
        Compute normals of each tensors reference point and create a property

        :param tensors_property: the tensors from which the normals will be computed
        :param min_points: minimum points required to compute normal for a point

        :return: normals property for the tensors

        :rtype: NormalsProperty
        """
        normals = list(map(lambda t: t.plate_axis, tensorProperty))
        return NormalsProperty(tensorProperty.Points, np.asarray(normals))

    @classmethod
    def normal_from_tensors_with_CUDA(cls, neighborProperty):
        """
        Compute normals of each point using CUDA
        :param neighborProperty: neighborProperty to compute normal for each of its points

        :type neighborProperty: NeighborProperty

        :return:NeighborProperty ,normals (GPU use) shape (3*n,) while n is the number of normals

        :rtype: NormalsProperty

        **Usage example**

        .. literalinclude:: ../../NormalsFactory.py
            :lines: 358-365
            :linenos:

        """
        pnts = neighborProperty.Points.ToNumpy()
        cudaNeighbors, numNeighbors = neighborProperty.ToCUDA
        numNeighbors = numNeighbors.reshape((-1, 1))
        print("start gpu normals")
        start = timer()
        normals = compute_normals_by_tensor_cuda(pnts, numNeighbors, cudaNeighbors)
        duration = timer() - start
        print("gpu : ", duration)
        return NormalsProperty(neighborProperty.Points, normals.reshape((-1, 3))), normals

    @staticmethod
    def normals_from_file(points, normalsFileName):
        """
        Loads normals from file

        :param points: PointSet to which the normals belong to
        :param normalsFileName: name of file containing normal for each points

        :type points: PointSet.PointSet
        :type normalsFileName: str

        :return: NormalsProperty of the points

        :rtype: NormalsProperty
        """

        # Read normals from file                 
        parametersTypes = dtype({'names': ['dx', 'dy', 'dz']
                                    , 'formats': ['float', 'float', 'float']})

        imported_array = genfromtxt(normalsFileName, dtype=parametersTypes, filling_values=(0, 0, 0))
        dxdydz = imported_array[['x', 'y', 'z']].view(float).reshape(len(imported_array), -1)
        normals = NormalsProperty(points, dxdydz)

        return normals

    @staticmethod
    def normalsPCA(neighborsProperty, viewpoint = np.array([0,0,30])):
        """
        Computes the normals for each point in the pointset via PCA

        :param neighborsProperty: a neighborhood property from which the PCA is computed

        :type neighborsProperty: Properties.Neighborhood.NeighborsProperty.NeighborsProperty

        :return: normals for each point
        
        :rtype: NormalsProperty

        """
        normals = []
        for neighborhood in neighborsProperty:
            current_normal = NormalsFactory.__normal_perPoint_PCA(neighborhood)
            if current_normal.dot(viewpoint - neighborhood.center_point_coords) < 0:
                current_normal = -current_normal
            if current_normal[2] < 0:
                # print(neighborhood.center_point_idx)
                current_normal = -current_normal #TODO: this is not correct for all situations, but a temporary patch

            if neighborhood.numberOfNeighbors < 3:
                print(neighborhood.center_point_idx)
                current_normal = np.array([0, 0, 0])
            normals.append(current_normal)

        return NormalsProperty(neighborsProperty.Points, np.array(normals))

    @staticmethod
    def __normal_perPoint_PCA(point_neighbors):
        r"""
        Compute the normal at each point according to its neighbors, via PCA

        Using pt as the point where the normal should be computed, the vectors :math:`y` from it are computed

        .. math::

            y_i = x_i - pt

        minimizing

        .. math::

            \min_{||{\bf n}||=1} \sum_{i=1}^n\left({\bf y}_i^T{\bf n}\right)^2

        we get:

        .. math::

            \begin{eqnarray}
            f({\bf n}) = {\bf n}^T{\bf Sn} \qquad ({\bf S}={\bf YY^T} \\
            \min f({\bf n})  \quad s.t. {\bf n}^T{\bf n})=1

        Using Lagrange multipliers we get:

        .. math::

            {\bf Sn|=\lambda {\bf n}

        which means that :math:`{\bf n}` is the eigenvector of :math:`{\bf S}` with the smallest eigenvalue

        :param point_neighbors: the point neighborhood

        :type point_neighbors: Properties.Neighborhood.PointNeighborhood.PointNeighborhood

        :return: normal to the point

        :rtype: np.ndarray
        """

        y = point_neighbors.neighbors_vectors()
        # try:
        #     u, s, vh = np.linalg.svd(y.T)
        #     return u[:, -1]
        # except:
        eigval, eigvec = np.linalg.eig(y.T.dot(y))
        return eigvec[:, np.argmin(eigval)]

    @classmethod
    def normals_panorama_xyz(cls, panorama, ksize=3, resolution=1, sigma=0,**kwargs):
        r"""
        Compute normals in panorama, after adaptive smoothing according to :cite:`Arav2013` and  :cite:`Zeibak2008`.

        :param panorama: the panorama via which the normals are computed
        :param ksize: filter kernel size (1, 3, 5, or 7). Default: 3
        :param resolution: kernel resolution. Default: 1
        :param sigma: sigma for gaussian blurring. Default: 0. If sigma=0 no smoothing is carried out

        :type panorama: Properties.Panoramas.PanoramaProperty.PanoramaProperty

        :return: normals a holding  :math:`n\times m \times 3` ndarray of the normals in each direction (Nx, Ny, Nz)

        :rtype: np.ndarray [:math:`n\times m \times 3`]


        the normals are computed as

        .. math::
            \vec{v}_1 = \begin{bmatrix} dX_1\\ dY_1 \\dZ_1 \end{bmatrix} ; \qquad
            \vec{v}_2 = \begin{bmatrix} dX_2\\ dY_2 \\dZ_2 \end{bmatrix}

        with :math:`dX_i` etc. the differentiation in each direction. Then,

        .. math::
            \vec{N} = \frac{\vec{v}_1\times \vec{v}_2}{||\vec{v}_1\times \vec{v}_2||}

        .. warning::
           Implemented for gaussian filtering and adaptive filters. Other adaptations might be
           needed when using different methods for smoothing.

        """
        import MyTools as mt

        img = panorama.panoramaImage

        xi, yi = panorama.pano2rad()

        # Computing Normal vectors
        #---------------------------

        # finding the ray direction (Zeibak Thesis: eq. 20, p. 58)
        x = img * np.cos(yi) * np.cos(xi)
        y = img * np.cos(yi) * np.sin(xi)
        z = img * np.sin(yi)

        # Local derivatives (according to Zeibak p. 56)
        dfx_daz, dfx_delevation = mt.computeImageDerivatives(x, order=1, ksize=ksize, resolution=resolution, sigma=sigma)
        dfy_daz, dfy_delevation = mt.computeImageDerivatives(y, order=1, ksize=ksize, resolution=resolution, sigma=sigma)
        dfz_daz, dfz_delevation = mt.computeImageDerivatives(z, order=1, ksize=ksize, resolution=resolution, sigma=sigma)

        v1 = np.zeros((x.shape[0], x.shape[1], 3))
        v2 = np.zeros((x.shape[0], x.shape[1], 3))

        v1[:, :, 0] = dfx_delevation
        v1[:, :, 1] = dfy_delevation
        v1[:, :, 2] = dfz_delevation

        v2[:, :, 0] = dfx_daz
        v2[:, :, 1] = dfy_daz
        v2[:, :, 2] = dfz_daz

        cross_vec = np.cross(v1, v2, axis=2)
        n = cross_vec / np.linalg.norm(cross_vec, axis=2)[:, :, None]

        return n

    @staticmethod
    def normals_open3D(pointcloud, search_radius=0.05, maxNN=200, orientation=(0., 0., 0.), return_pcl = False):
        """
        Computes the normals using open 3D

        :param pointcloud: an open 3d point cloud object
        :param search_radius: neighbors radius for normal computation. Default: 0.05
        :param maxNN: maximum neighbors in a neighborhood. If set to (-1), there is no limitation. Default: 20.
        :param orientation: "camera" orientation. The orientation towards which the normals are computed. Default: (0,0,0)
        :param return_pcl: a flag to return the computed point cloud with the normals (for o3d visualization). Default: False

        :type pointcloud: DataClasses.BaseData.BaseData
        :type search_radius: float
        :type maxNN: int
        :type orientation: tuple
        :type return_pcl: bool

        :return: normals property and the pointcloud with normals
        
        :rtype: NormalsProperty
        
        .. warning::
            The computed normals act weird on edges. Pay attention.
        """
        # checking if the set point set is an object of PointSetOpen3D
        if not isinstance(pointcloud, PointSetOpen3D):
            _pointcloud = PointSetOpen3D(pointcloud)
        else:
            _pointcloud = pointcloud

        _pointcloud.CalculateNormals(search_radius, maxNN, orientation)  # computing the normals using open3D method
        normals = np.array(_pointcloud.data.normals)
        if return_pcl:
            return NormalsProperty(pointcloud, normals), _pointcloud
        else:
            return NormalsProperty(pointcloud, normals)

        # TODO: obsolete code for computing normals using vtk, should probably be deleted
        # @staticmethod
        # def __CalcAverageNormal(x, y, z, normalsPoints, normals, eps=0.00001):
        #
        #     indices = nonzero(sum((normalsPoints - [x, y, z]) ** 2, axis=-1) < eps ** 2)[0]
        #     return mean(normals[indices], axis=0)
        #
    # @staticmethod
    # def VtkNormals(points, triangulation=None):
    #     """
    #     Calculate normals for each points as average of normals of trianges to which the points belongs to.
    #     If no triangulation is given, use TriangulationFactory.Delaunay2D
    #
    #     :Args:
    #
    #         - points: PointSet/PointSubSet object
    #         - triangulation: triangulationProperty
    #
    #
    #
    #     :Returns:
    #         - NormalsProperty
    #     """
    #     polyData = points.ToPolyData
    #
    #     if triangulation == None:
    #         triangulation = TriangulationFactory.Delaunay2D(points)
    #
    #     polyData.polys = triangulation.TrianglesIndices()
    #
    #     compute_normals = mlab.pipeline.poly_data_normals(polyData)
    #
    #     #        normals = compute_normals.outputs[0].point_data.normals.to_array()
    #
    #     mlab.close()
    #
    #     normals = asarray(map(partial(NormalsFactory.__CalcAverageNormal,
    #                                   normalsPoints=compute_normals.outputs[0].points.to_array(),
    #                                   normals=compute_normals.outputs[0].point_data.normals.to_array()), points.X,
    #                           points.Y, points.Z))
    #     normals = compute_normals.outputs[0].point_data.normals.to_array()[0: points.Size]
    #
    #     return NormalsProperty(points, normals)


#

if __name__ == "__main__":
    # TODO: Obsolete code, should be deleted or modified to a newer version
    from Properties.Neighborhood.NeighborsFactory import NeighborsFactory
    from Properties.Curvature.CurvatureFactory import CurvatureFactory
    from Properties.Curvature.CurvatureProperty import CurvatureProperty
    from DataClasses.KdTreePointSet import KdTreePointSet
    from DataClasses.PointSubSetOpen3D import PointSetOpen3D
    from DataClasses.PointSet import PointSet
    from VisualizationO3D import VisualizationO3D
    from timeit import default_timer as timer

    points = np.loadtxt("/home/user/PycharmProjects/Filin-Infrastructure/test_data/tigers1M.txt")
    pnts = points[:, :3]
    pntSet = PointSet(pnts)
    # pntSet = IOFactory.ReadFunctions.ReadPts("/home/user/PycharmProjects/Filin-Infrastructure/test_data/test_pts.pts")
    neiProp = NeighborsFactory.kdtreePointSet_knn(KdTreePointSet(pntSet), 500)
    # normals = np.array(list(map(lambda i: computeNormalByTensor(pntSet[ind[i, 1:], :], pntSet[i]), range(numPnts))))
    # neiProp2Bool = np.array(list(map(lambda i: CurvatureFactory.checkNeighborhood(neiProp), range(pntSet.Size))))
    # for p in neiProp :
    #     CurvatureFactory.checkNeighborhood(p)
    # neiProp2Bool = CurvatureFactory.checkNeighborhood(neiProp)
    normalsProp, normals = NormalsFactory.normal_from_tensors_with_CUDA(neiProp)
    curv = CurvatureFactory.curvature_with_CUDA(neiProp, normals)

    p3d = PointSetOpen3D(pntSet)

    print("start cpu normals")
    start = timer()
    p3d.CalculateNormals(1, maxNN=500)
    duration = timer() - start
    print("cpu : ", duration)
    open3d_normals = np.array(p3d.data.normals)

    v3d = VisualizationO3D()
    # gpu_normals = normalsProp.Normals
    # dot_pro = np.dot(gpu_normals, open3d_normals.T).diagonal()
    # v3d.visualize_property(normalsProp)

    # p3d.data.normals = o3d.Vector3dVector(normalsProp.Normals)
    # v3d.visualize_pointset(p3d)
    # normals_o3d = np.asarray(o3d.data.normals)

    search_radius = 0.25
    max_nn = 20

    normals0 = NormalsProperty(p3d, normals.reshape((-1, 3)))
    normals1 = NormalsProperty(p3d, np.asarray(p3d.data.normals))

    # neighborsProperty = NeighborsFactory.pointSetOpen3D_knn_kdTree(p3d, max_nn)
    curvatures = CurvatureFactory.umbrella_curvature(neiProp, normals1, valid_sectors=4, invalid_value=0,
                                                     verbose=True)

    curvatures1 = CurvatureProperty(neiProp.Points, principal_curvatures=None,
                                    umbrella_curvature=curv)

    v3d.visualize_property(curvatures)
    v3d.visualize_property(curvatures1)


    print("done")
