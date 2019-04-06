'''
|today|
'''

import numpy as np
from tqdm import tqdm

import RotationUtils
from CurvatureProperty import CurvatureProperty
from RasterData import RasterData


class CurvatureFactory:
    '''
    curvature parameters computation
    '''

    @classmethod
    def tensorProperty_3parameters(cls, tensorProperty, min_points=5):
        r"""
        Compute curvature for tensors, via the *three-parameters* algorithm.

        Curvature for tensors that have less points than min_points will not be computed.

        :param tensorProperty: the tensors of the points cloud (precomputed)
        :param min_points: minimum points required to compute curvature for a point

        :type tensorProperty: TensorProperty.TensorProperty

        :return: Curvature property

        :rtype: CurvatureProperty

        .. seealso::

            :meth:`curvature_by_3parameters`
        """
        k1 = []
        k2 = []
        # Find normal to tensors
        for tensor in tensorProperty:
            num_pts = tensor.points_number
            if num_pts > min_points:
                k1_, k2_ = cls.curvature_by_3parameters(tensor.points, tensor.normal())
                k1.append(k1_)
                k2.append(k2_)

        k1 = np.asarray(k1)
        k2 = np.asarray(k2)

        return CurvatureProperty(tensorProperty.Points, np.vstack((k1, k2)).T)

    @classmethod
    def pointSetOpen3D_3parameters(cls, pointset3d, neighbors_property,
                                   min_points_in_neighborhood=5, min_points_in_sector=2, valid_sectors=7, num_sectors=8,
                                   invalid_value=-999, verbose=False):
        """
        Curvature of a PointSetOpen3D with its normals and neighborhood via the *three-parameters* algorithm.

        .. warning::

            If a pointset3D is sent without its normals computed in advance, the normals will be computed according to the first
            point neighborhood parameters

        :param pointset3d: the point cloud
        :param neighbors_property: the neighbors property of the point cloud
        :param min_points_in_neighborhood: minimal number of points in a neighborhood to make it viable for curvature computation. Default: 5
        :param min_points_in_sector: minimal points in a sector to be considered valid. Default: 2
        :param valid_sectors: minimal sectors needed for a point to be considered good. Default: 7
        :param num_sectors: the number of sectors the circle is divided to. Default: 8
        :param invalid_value: value for invalid curvature (points that their curvature cannot be computed). Default: -999
        :param verbose: print inter running messages. default: False

        :type pointset3d: PointSetOpen3D.PointSetOpen3D
        :type neighbors_property: NeighborsProperty.NeighborsProperty
        :type min_points_in_neighborhood: int
        :type min_points_in_sector: int
        :type valid_sectors: int
        :type num_sectors: int
        :type invalid_value: float
        :type verbose: bool

        :return: curvature property

        :rtype: CurvatureProperty

        .. seealso::

            :meth:`curvature_by_3parameters`
        """
        from PointSetOpen3D import PointSetOpen3D
        from PointNeighborhood import PointNeighborhood
        k1 = []
        k2 = []
        print('>>> Compute all points curvatures using the 3-parameters algorithm')
        if not isinstance(pointset3d, PointSetOpen3D):
            import warnings
            warnings.warn('Pointset must be PointSetOpen3D')
            return 1

        # check if normals have been computed before for this point cloud:
        if np.asarray(pointset3d.data.normals).shape[0] == 0:

            for neighborhood1 in neighbors_property:
                radius = neighborhood1.radius
                maxNN = neighborhood1.numberOfNeighbors
                if radius is not None and maxNN is not None:
                    pointset3d.CalculateNormals(search_radius=radius, maxNN=maxNN)
                    break

        normals = np.asarray(pointset3d.data.normals)
        neighbors_property.__reset__()  # reset iterable

        for neighborhood, i in zip(neighbors_property, range(pointset3d.Size)):
            if i == 578:
                print(i)
            if isinstance(neighborhood, PointNeighborhood) and cls.__checkNeighborhood(neighborhood,
                                                                                       min_points_in_neighborhood=min_points_in_neighborhood,
                                                                                       min_points_in_sector=min_points_in_sector,
                                                                                       valid_sectors=valid_sectors,
                                                                                       num_sectors=num_sectors):
                normal = normals[i, :]
                k1_, k2_ = cls.curvature_by_3parameters(neighborhood, normal)
                k1.append(k1_)
                k2.append(k2_)

                if verbose:
                    print(k1_, k2_)

            else:
                print(i)
                k1.append(invalid_value)
                k2.append(invalid_value)
        k1 = np.asarray(k1)
        k2 = np.asarray(k2)

        curvatures = CurvatureProperty(pointset3d, np.vstack((k1, k2)).T)
        curvatures.set_invalid_value(invalid_value)
        return curvatures

    @classmethod
    def curvature_by_3parameters(cls, neighborhood, normal):
        r"""
        Curvature computation based on fundamental form when fitting bi-quadratic surface.

        :param neighborhood: the neighbors according to which the curvature is computed
        :param normal: normal to the point which the curvature is computed

        :type neighborhood: PointNeighborhood.PointNeighbohhod, np.ndarray, PointSet.PointSet
        :type normal: np.array

        :return: minimal and maximal curvature

        :rtype: tuple

        **Algorithm**

        #. Rotate so that the normal will be [0 0 1]
            the bi-quadratic surface coefficients :math:`d,e,f` are zero and the fitting is of three parameters:

            .. math::

                z(x,y) = ax^2 + by^2 + cxy

        #. Second derivatives dictate the curvature

           .. math::
               {\bf H} = \begin{bmatrix} 2a & c \\ c & 2b \end{bmatrix}

        #. The principal curvatures are computed as:

            .. math::
                C=\sqrt{c^2+\left[\left(a-b\right)\right]^2}

            .. math::
                \kappa_{1,2} = \frac{a+b}{2}\pm C

        """
        from PointNeighborhood import PointNeighborhood
        from PointSet import PointSet

        if isinstance(neighborhood, PointNeighborhood):
            neighbors = neighborhood.neighbors.ToNumpy()

        elif isinstance(neighborhood, PointSet):
            neighbors = neighborhood.ToNumpy()

        elif isinstance(neighborhood, np.ndarray):
            neighbors = neighborhood
        else:
            import warnings
            warnings.warn('Curvature_3d_params: Unknown type of neighborhood')
            return 1

        pnt = neighbors[0, :]
        # remove reference point from neighbors array
        neighbors = neighbors[1:, :]

        # if a normal of a neighborhood is in an opposite direction rotate it 180 degrees
        if np.linalg.norm(pnt, 2) < np.linalg.norm(pnt + normal, 2):
            normal = -normal
        n = np.array([0, 0, 1])

        # rotate the neighborhood to xy plane
        rot_mat = RotationUtils.Rotation_2Vectors(normal, n)

        neighbors = (np.dot(rot_mat, neighbors.T)).T

        # compute curvature by 3 parameters
        p = CurvatureFactory.__BiQuadratic_Surface(neighbors)
        a = p[0]
        b = p[1]
        c = p[2]

        C = np.sqrt(c ** 2 + (0.5 * (a - b)) ** 2)
        k1 = (a + b) / 2 + C
        k2 = (a + b) / 2 - C

        return k1[0], k2[0]

    @classmethod
    def umbrella_curvature(cls, neighbrohood, normals,
                           min_points_in_neighborhood=8, min_points_in_sector=1, valid_sectors=7, num_sectors=8,
                           invalid_value=-999, cuvatureProperty=None, verbose=False):
        r"""
        Compute an umbrella curvature,

        Defined in :cite:`Foorginejad.Khalili2014`

        Based on the idea that the size of the curvature (openness) is influenced by the projection of the neighboring points on the normal vector.

        .. math::
            \kappa_{um} = \sum_{i=1}^m \left| \frac{N_i-p}{\left|N_i-p\right|}\cdot \bf{n}\right|

        with :math:`N_i` the neighbor i of point :math:`p`, and :math:`\bf{n}` the normal at point :math:`p`.

        :param neighbrohood: neighborhood property of a point cloud
        :param normals: normals property of a point cloud
        :param min_points_in_neighborhood: minimal number of points in a neighborhood to make it viable for curvature computation. Default: 8
        :param min_points_in_sector: minimal points in a sector to be considered valid. Default: 1
        :param valid_sectors: minimal sectors needed for a point to be considered good. Default: 7
        :param num_sectors: the number of sectors the circle is divided to. Default: 8
        :param invalid_value: value for invalid curvature (points that their curvature cannot be computed). Default: -999
        :param cuvatureProperty: A curvature property to update with the umbrella curvature. If None, a new curvature property will be created, with the principal curvatures as None.
        :param verbose: print inter running messages. default: False

        :type neighbrohood: NeighborsProperty.NeighborsProperty
        :type normals: NormalsProperty.NormalsProperty
        :type min_points_in_neighborhood: int
        :type min_points_in_sector: int
        :type valid_sectors: int
        :type num_sectors: int
        :type invalid_value: float
        :type cuvatureProperty: CurvatureProperty
        :type verbose: bool

        :return: umbrella curvature for all points that have enough neighbors for the curvature to be computed.

        :rtype: np.array

        .. seealso::

           :meth:`__good_point` and :meth:`__checkNeighborhood`

        """
        umbrellaCurvature = []

        for point_neighbors in tqdm(neighbrohood, total=neighbrohood.Size,
                                    desc='Compute umbrella curvature for all point cloud'):
            # check if the neighborhood is valid
            if cls.__checkNeighborhood(point_neighbors, min_points_in_neighborhood=min_points_in_neighborhood,
                                       min_points_in_sector=min_points_in_sector, valid_sectors=valid_sectors,
                                       num_sectors=num_sectors):
                point_idx = point_neighbors.neighborhoodIndices[0]
                n = normals.Normals[point_idx]

                if verbose:
                    print(point_idx, n)

                # compute the directions projections  on the normal at the center point of each neighbor
                directions = point_neighbors.neighbors_vectors().dot(n)
                umbrellaCurvature.append(np.sum(directions))

            else:
                if verbose:
                    print('invalid point:', point_neighbors.center_point_idx)
                umbrellaCurvature.append(invalid_value)
        umbrella_curvature = np.asarray(umbrellaCurvature)
        if cuvatureProperty is None:
            return CurvatureProperty(neighbrohood.Points, principal_curvatures=None,
                                     umbrella_curvature=umbrella_curvature)
        else:
            cuvatureProperty.load(None, umbrella_curvature=umbrella_curvature)


    @classmethod
    def raster_fundamentalForm(cls, raster, ksize=3, sigma=2.5, gradientType='L1'):
        """
        Compute raster curvature based on first fundamental form

        :param raster: the raster image
        :param ksize: window/kernel size (equivalent to neighborhood size)
        :param sigma: for Gaussian blurring
        :param gradientType: 'L1' or 'L2' for derivatives computation

        :type raster: np.array or RasterData
        :type window_size: int
        :type sigma: float
        :type gradientType: str

        :return: curvature map

        :rtype: curvatureProperty

        """
        import MyTools as mt
        import cv2

        if isinstance(raster, RasterData):
            img = raster.data
            voidData = raster.voidData
        else:
            img = raster
            voidData = -9999

        img[img == voidData] = np.mean(np.mean(img[img != voidData]))

        img = cv2.normalize(img.astype('float'), None, 0.0, 1.0,
                            cv2.NORM_MINMAX)  # Convert to normalized floating point

        dx, dy, Zxx, Zyy, Zxy = mt.computeImageDerivatives(img, 2, sigma=sigma,
                                                           gradientType=gradientType,
                                                           ksize=ksize)

        k1 = (((Zxx + Zyy) + np.sqrt((Zxx - Zyy) ** 2 + 4 * Zxy ** 2)) / 2)
        k2 = (((Zxx + Zyy) - np.sqrt((Zxx - Zyy) ** 2 + 4 * Zxy ** 2)) / 2)

        if isinstance(raster, RasterData):
            return CurvatureProperty(raster, np.concatenate((k1[:, :, None], k2[:, :, None]), axis=2))
        else:
            return np.concatenate((k1[:, :, None], k2[:, :, None]), axis=2)

    # ------------------ PRIVATE METHODS ----------------------------
    @classmethod
    def __checkNeighborhood(cls, neighborhood, min_points_in_neighborhood=5, min_points_in_sector=2, valid_sectors=7,
                            num_sectors=4):
        """
        Check if a neighborhood is viable for curvature computation.

        Decided according to number of neighbors and distribution of the points around the reference points.
        The second condition is defined by the number of sectors with a minimum number of points.

        :param neighborhood: all points that compose the neighborhood of index 0 point
        :param min_points_in_neighborhood: minimal number of points in a neighborhood to make it viable for curvature computation. Default: 5
        :param min_points_in_sector: minimal points in a sector to be considered valid. Default: 2
        :param valid_sectors: minimal sectors needed for a point to be considered good. Default: 7
        :param num_sectors: the number of sectors the circle is divided to

        :type neighborhood: PointNeighborhood.PointNeighborhood
        :type min_points_in_neighborhood: int
        :type min_points_in_sector: int
        :type valid_sectors: int
        :type num_sectors: int
        :return: true if the neighborhood is valid; false otherwise

        :rtype: bool
        """

        # Check minimal number of neighbors
        # the numberOfNeighbors is with the reference points
        if neighborhood.numberOfNeighbors - 1 < min_points_in_neighborhood:
            return False

        # Check distribution
        neighborhood_array = neighborhood.neighbors.ToNumpy()
        return cls.__good_point(neighborhood_array, neighborhood_array[0, :], min_points_in_sector, valid_sectors,
                                num_sectors)

    @staticmethod
    def __good_point(neighbors, pnt=np.array([0, 0, 0]), min_points_in_sector=2, valid_sectors=7, num_sectors=4):
        '''
        Determine whether the point is appropriate for curvature calculation according to the spread of its neighbours.

        Dividing the neighborhood into 8 sectors around the point. If there are more than ``min_points_in_sector``, i.e.,
        more than the minimum number of points within that sector, and there are more than ``valid_sectors`` that answer
        that condition -- the point is considered good for curvature computation.

        :param neighbors: neighboring points.
        :param pnt: the point to which the curvature should be computed. Default: [0,0,0]
        :param min_points_in_sector: minimal points in a sector to be considered valid. Default: 2
        :param valid_sectors: minimal sectors needed for a point to be considered good. Default: 7
        :param num_sectors: the number of sectors the circle is divided to

        :type neighbors: PointSet, numpy.array
        :type pnt: numpy.array
        :type min_points_in_sector: int
        :type valid_sectors: int
        :type num_sectors: int

        1. Calculate Azimuth between the reference point and the points in neighborhood
        2. Check number of points in every 45 degree sector

        :return: True if the point is good, False otherwise

        :rtype: bool
        '''
        if min_points_in_sector is None:
            min_points_in_sector = 2

        if valid_sectors is None:
            valid_sectors = 7

        if num_sectors is None:
            num_sectors = 4



        if isinstance(neighbors, np.ndarray):
            neighbor_points = neighbors
        else:
            neighbor_points = neighbors.ToNumpy()

        neighbor_points = neighbor_points - pnt.flatten()

        count_valid_sectors = 0
        p_angle = np.zeros((1, neighbor_points.shape[0]))

        # Calculate Azimuth with neighbors who do not have X coordinate zero
        ind1 = np.where(np.abs(neighbor_points[:, 0]) > 1e-6)[0]
        p_angle[0, ind1] = np.arctan2(neighbor_points[ind1, 1], neighbor_points[ind1, 0])

        # Calculate Azimuth ith neighbors who have X coordinate zero
        ind2 = np.where(np.abs(neighbor_points[:, 0]) <= 1e-6)[0]
        if ind2.size != 0:
            ind2_1 = np.where(neighbor_points[ind2, 1] > 0)[0]
            ind2_2 = np.where(neighbor_points[ind2, 1] < 0)[0]
            if ind2_1.size != 0:
                p_angle[0, ind2[ind2_1]] = np.pi / 2.0
            if ind2_2.size != 0:
                p_angle[0, ind2[ind2_2]] = 3.0 * np.pi / 2.0

        p_angle[np.where(p_angle < 0)] += 2 * np.pi

        # divide the circle into sectors: starting from zero, reaching to 270deg + 45deg.
        for i in np.linspace(0, 3 * np.pi / 2, num_sectors):
            p_in_sector = (np.where(p_angle[np.where(p_angle <= i + np.pi / int(num_sectors / 2))] >= i))[0].size
            if p_in_sector >= min_points_in_sector:
                count_valid_sectors += 1

        if count_valid_sectors >= valid_sectors:
            return True
        else:
            return False

    @staticmethod
    def __keep_good_curves_data(curves_data, pointset_open3D):
        """
        Filters out points and their curvature that their curvature was not computed.

        :param curves_data: an array of curvature data
        :param pointset_open3D: the point set that relates to the curvature data

        :type curves_data: np.ndarray
        :type pointset_open3D: PointSetOpen3D.PointSetOpen3D

        :return: the curvature data and the pointset without points that their curvature wasn't calculated

        :rtype: CurvatureProperty
        """
        ind_not_good = np.where(curves_data == -999)[0]
        ind_not_good = np.unique(ind_not_good, return_index=True)[0]
        curves_data = np.delete(curves_data, ind_not_good, axis=0)
        pointset_open3D.DisregardPoints(ind_not_good)

        # curves_data = FilterBySTD(curves_data, pointsetExtra, maxSTD=1., minSTD=0.4)

        k1 = np.expand_dims(curves_data[:, 0], 1)
        k2 = np.expand_dims(curves_data[:, 1], 1)
        curves = CurvatureProperty(pointset_open3D, np.hstack((k1, k2)))
        return curves

    @staticmethod
    def __BiQuadratic_Surface(neighbor_points):
        '''
        BiQuadratic surface adjustment to discrete point cloud

        :param neighbor_points: 3D points coordinates

        :type neighbor_points: nx3 array

        :return: p - surface's coefficients

        :rtype: nd-array

        '''
        # ==== initial guess ====
        x = np.expand_dims(neighbor_points[:, 0], 1)
        y = np.expand_dims(neighbor_points[:, 1], 1)
        z = np.expand_dims(neighbor_points[:, 2], 1)

        A = np.hstack((x ** 2, y ** 2, x * y))
        N = A.T.dot(A)
        u = A.T.dot(z)
        p = np.linalg.solve(N, u)

        return p

