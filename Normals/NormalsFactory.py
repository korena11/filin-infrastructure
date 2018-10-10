# import mayavi.mlab as mlab
import sys
from warnings import warn

import numpy as np
from numpy import dtype, genfromtxt, nonzero, mean, sum
from sklearn.neighbors import BallTree, KDTree

from Normals.NormalsProperty import NormalsProperty
from PointSet import PointSet

if sys.platform == 'linux':
    pass


class NormalsFactory:
    @staticmethod
    def normals_from_file(points, normalsFileName):
        """

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
    def normalsPCA(pointset, tree=None, radius=None, k_neighbors=None, **kwargs):
        """
        Computes the normals for each point in the pointset via PCA

        :param pointset: points to compute their normals
        :param tree: a KD-tree or Ball-tree to extract neighbors from. If no tree is passed, a ball-tree will be
        build here.

        *One of these should be passed*

        :param radius: the radius in which the neighbors should be found
        :param k_neighbors: number of neighbors

        **Optionals**

        :param leaf_size: for ball-tree construction. Default: 40
        :param metric: for ball-tree construction. Deafault: 'minkowski'

        :type pointset: PointSet
        :type tree: BallTree or KDTree
        :type radius: float
        :type k_neighbors: int
        :type leaf_size: int
        :type metric: str

        :return: normals for each point
        
        :rtype: NormalsProperty

        """
        leaf_size = kwargs.get('leaf_size', 40)
        metric = kwargs.get('metric', 'minkowski')
        normals = []

        # Build tree if no tree was passed
        if tree is None:
            tree = BallTree(pointset.ToNumpy(), leaf_size=leaf_size, metric=metric)
        points = pointset.ToNumpy()
        for pt in points:

            # first try by radius
            try:
                neighbors_ind = tree.query_radius([pt], r=radius)

            # if no radius given, try by number of neighbors
            except TypeError:
                try:
                    dist, neighbors_ind = tree.query([pt], k=k_neighbors)
                except TypeError:
                    warn('Size of radius or number of neighbors are required')
                    return -1

            normals.append(NormalsFactory.__normal_perPoint_PCA(pt, points[neighbors_ind[0].astype(int), :]))

        return NormalsProperty(pointset, np.array(normals))

    @staticmethod
    def __normal_perPoint_PCA(pt, neighbors):
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

        :param pt: x,y,z point
        :param neighbors: [x,y,z] points that were selected as neighbors

        :type pt: np.ndarray
        :type neighbors: np.ndarray

        :return: normal to the point

        :rtype: np.ndarray
        """

        y = neighbors - pt

        # try:
        #     u, s, vh = np.linalg.svd(y.T)
        #     return u[:, -1]
        # except:
        eigval, eigvec = np.linalg.eig(y.T.dot(y))
        return eigvec[:, np.argmin(eigval)]

    @staticmethod
    def normalsComputation_in_raster(x, y, z):
        r"""
        Computes the normal vectors according where each ordinate is ordered at its place as an ndarray (mxn)
        Usually used for computing normals in panorama images or other rasters.

        According to :cite:`Zeibak2008`, the normals are computed as

        .. math::
            \vec{v}_1 = \begin{bmatrix} dX_1\\ dY_1 \\dZ_1 \end{bmatrix} ; \qquad
            \vec{v}_2 = \begin{bmatrix} dX_2\\ dY_2 \\dZ_2 \end{bmatrix}

        with :math:`dX_i` etc. the differentiation in each direction. Then,

        .. math::

            \vec{N} = \frac{\vec{v}_1\times \vec{v}_2}{||\vec{v}_1\times \vec{v}_2||}

        :param x: x ordinates as organized in the raster
        :param y: y ordinates as organized in the raster
        :param z: z ordinates as organized in the raster

        :return: normals matrices for each direction

        :rtype: np.ndarray [:math:`n\times m \times 3`]

        """
        import MyTools as mt
        # Local derivatives (according to Zeibak p. 56)
        dfx_daz, dfx_delevation = mt.computeImageDerivatives(x, order=1)
        dfy_daz, dfy_delevation = mt.computeImageDerivatives(y, order=1)
        dfz_daz, dfz_delevation = mt.computeImageDerivatives(z, order=1)

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
    def normals_open3D(pointcloud):
        """
        Computes the normals using open 3D

        :param pointcloud: an open 3d point cloud object
        :type pointcloud: open3d.PointCloud

        :return: normals property and the pointcloud with normals
        """

    @staticmethod
    def __CalcAverageNormal(x, y, z, normalsPoints, normals, eps=0.00001):

        indices = nonzero(sum((normalsPoints - [x, y, z]) ** 2, axis=-1) < eps ** 2)[0]
        return mean(normals[indices], axis=0)

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
    from IOFactory import IOFactory
    from Visualization import Visualization

    pointSetList = []

    #    IOFactory.ReadXYZ('..\\Sample Data\\cubeSurface.xyz', pointSetList)
    #    normalsFileName = '..\\Sample Data\\cubeSurfaceNormals.xyz'
    #    normals = NormalsFactory.ReadNormalsFromFile(pointSetList[0], normalsFileName)
    IOFactory.ReadXYZ(r'D:\\Documents\\Pointsets\\cylinder_1.3_Points.txt', pointSetList)
    #    triangulation = TriangulationFactory.Delaunay2D(pointSetList[0])
    normals = NormalsFactory.VtkNormals(pointSetList[0])  # , triangulation)

    Visualization.RenderPointSet(normals, 'color', color=(0, 0, 0), pointSize=3)
    Visualization.Show()

#    points3d(pointSetList[0].X(), pointSetList[0].Y(), pointSetList[0].Z(), scale_factor=.25)
#    quiver3d(pointSetList[0].X(), pointSetList[0].Y(), pointSetList[0].Z(), normals.dX(), normals.dY(), normals.dZ())    
#    show()
