'''
|today|
'''

import numpy as np
from tqdm import tqdm

from Cuda.cuda_API import *
import Properties.Transformations.RotationUtils as RotationUtils
from DataClasses.RasterData import RasterData
from Properties.Curvature.CurvatureProperty import CurvatureProperty


class CurvatureFactory:
    '''
    curvature parameters computation
    '''

    @classmethod
    def curvature_with_CUDA(cls, neighborProperty, normals):
        """
        Compute normals of each point using CUDA

        :param neighborProperty: neighborProperty to compute normal for each of its points
        :param normals: normals shape (3*n,) while n is the number of normals

        :type neighborProperty: NeighborProperty
        :type normals: np.array (3xn,)

        :return: curvature as 1 dim numpy array

         **Usage example**

        .. literalinclude:: ../../../../Properties/Normals/NormalsFactory.py
            :lines: 358-366
            :linenos:
        """
        pnts = neighborProperty.Points.ToNumpy()
        cudaNeighbors, numNeighbors = neighborProperty.ToCUDA
        numNeighbors = numNeighbors.reshape((-1, 1))
        print("start gpu curvature")
        start = timer()
        curv = computeUmbrelaCurvatureCuda(pnts, numNeighbors, cudaNeighbors, normals)
        duration = timer() - start
        print("gpu : ", duration)
        return curv

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
    def pointSetOpen3D_3parameters(cls, neighbors_property, normals_property=None,
                                   min_points_in_neighborhood=5, min_points_in_sector=2, valid_sectors=7, num_sectors=8,
                                   invalid_value=-999, alpha=0.05, verbose=False):
        """
        Curvature of a PointSetOpen3D with its normals and neighborhood via the *three-parameters* algorithm.

        .. warning::

            If a pointset3D is sent without its normals computed in advance, the normals will be computed according to the first
            point neighborhood parameters

        :param neighbors_property: the neighbors property of the point cloud
        :param normals_property: normals computed for the point cloud in advance
        :param min_points_in_neighborhood: minimal number of points in a neighborhood to make it viable for curvature computation. Default: 5
        :param min_points_in_sector: minimal points in a sector to be considered valid. Default: 2
        :param valid_sectors: minimal sectors needed for a point to be considered good. Default: 7
        :param num_sectors: the number of sectors the circle is divided to. Default: 8
        :param invalid_value: value for invalid curvature (points that their curvature cannot be computed). Default: -999
        :param verbose: print inter running messages. default: False

        :type pointset3d: PointSetOpen3D.PointSetOpen3D, DataClasses.BaseData.BaseData
        :type neighbors_property: NeighborsProperty.NeighborsProperty
        :type normals_property: NormalsProperty.NormalsProperty
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
        from DataClasses.PointSetOpen3D import PointSetOpen3D
        from Properties.Neighborhood.PointNeighborhood import PointNeighborhood
        pcl = neighbors_property.Points
        k1 = []
        k2 = []
        invalid_curvature = 0  # number of points that their curvature wasn't estimated
        print('\n >>> Estimate all points curvatures using the 3-parameters algorithm')
        if not isinstance(pcl, PointSetOpen3D):
            # import warnings
            # warnings.warn('Pointset must be PointSetOpen3D')
            # return 1
            pointset3d = PointSetOpen3D(pcl)
        else:
            pointset3d = pcl

        # check if normals have been computed before for this point cloud:
        if normals_property is None:
            if np.asarray(pointset3d.data.normals).shape[0] == 0:

                for neighborhood1 in neighbors_property:
                    if neighborhood1 is None:
                        continue
                    radius = neighborhood1.radius
                    maxNN = neighborhood1.numberOfNeighbors

                    if radius is not None and maxNN is not None:
                        pointset3d.CalculateNormals(search_radius=radius, maxNN=maxNN)

                        break

            normals = np.asarray(pointset3d.data.normals)
        else:
            normals = normals_property.Normals
        neighbors_property.__reset__()  # reset iterable

        for neighborhood, i in zip(neighbors_property, tqdm(range(pointset3d.Size), total=pointset3d.Size,
                                                            desc='Compute principal curvatures for all point cloud')):
            if isinstance(neighborhood, PointNeighborhood) and cls.__checkNeighborhood(neighborhood,
                                                                                       min_points_in_neighborhood=min_points_in_neighborhood,
                                                                                       min_points_in_sector=min_points_in_sector,
                                                                                       valid_sectors=valid_sectors,
                                                                                       num_sectors=num_sectors):
                if verbose:
                    if i == 2815:
                        print('!')
                normal = normals[i, :]
                k1_, k2_ = cls.curvature_by_3parameters(neighborhood, normal, alpha)

                if verbose:
                    print('k1 {};   k2 {}'.format(k1_, k2_))

                if np.abs(k1_ * k2_) > 1e3:
                    invalid_curvature += 1
                    k1.append(invalid_value)
                    k2.append(invalid_value)
                else:
                    k1.append(k1_)
                    k2.append(k2_)

            else:
                invalid_curvature += 1
                k1.append(invalid_value)
                k2.append(invalid_value)

        k1 = np.asarray(k1)
        k2 = np.asarray(k2)

        curvatures = CurvatureProperty(pcl, np.vstack((k1, k2)).T)
        curvatures.set_invalid_value(invalid_value)
        # print(invalid_curvature)
        return curvatures

    @classmethod
    def curvature_by_3parameters(cls, neighborhood, normal, alpha=0.05):
        r"""
        Curvature computation based on fundamental form when fitting bi-quadratic surface.

        :param neighborhood: the neighbors according to which the curvature is computed
        :param normal: normal to the point which the curvature is computed
        :param alpha: for hypothesis testing

        :type neighborhood: PointNeighborhood.PointNeighbohhod, np.ndarray, PointSet.PointSet
        :type normal: np.array
        :type alpha: float

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
        from Properties.Neighborhood.PointNeighborhood import PointNeighborhood
        from DataClasses.PointSet import PointSet

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

        # move points to center point
        neighbors -= neighbors[0, :]

        # if a normal of a neighborhood is in an opposite direction rotate it 180 degrees
        # if np.linalg.norm(pnt, 2) < np.linalg.norm(pnt + normal, 2):
        #     normal = -normal
        n = np.array([0, 0, 1])

        # rotate the neighborhood to xy plane
        rot_mat = RotationUtils.Rotation_2Vectors(normal, n)

        neighbors = (np.dot(rot_mat, neighbors.T)).T

        # compute curvature by 3 parameters
        p, Ftest_reject = CurvatureFactory.__BiQuadratic_Surface(neighbors, testCoeff=np.zeros((3,1)), alpha=alpha)
        # print(Ftest_reject)
        # if the assumption that the coefficients equal zero is not rejected, assign zeros coefficients
        if not Ftest_reject:
            p = np.zeros((3,1))

        a = 2*p[0]
        b = 2* p[1]
        c = p[2]

        C = np.sqrt(4*c ** 2 + ( (a - b)) ** 2)
        k1 = (a + b + C) / 2
        k2 = (a + b - C) / 2
        # print(neighbors.std(axis=0))
        if np.abs(k1 * k2) > 1e3:
            print('hello')
            print(neighbors.std(axis=0))
        return k1[0], k2[0]

    @classmethod
    def umbrella_curvature(cls, neighbrohood, normals, roughness=0.01, alpha=0.05,
                           min_points_in_neighborhood=5, min_points_in_sector=2, valid_sectors=7, num_sectors=8,
                           invalid_value=-999, curvatureProperty=None, verbose=False):
        r"""
        Compute an umbrella curvature,

        Defined in :cite:`Foorginejad.Khalili2014`

        Based on the idea that the size of the curvature (openness) is influenced by the projection of the neighboring points on the normal vector.

        .. math::
            \kappa_{um} = \sum_{i=1}^m \left| \frac{N_i-p}{\left|N_i-p\right|}\cdot \bf{n}\right|

        with :math:`N_i` the neighbor i of point :math:`p`, and :math:`\bf{n}` the normal at point :math:`p`.

        .. note::
            Each projection is checked so it will not be statistically zero in confidence level of :math:`\alpha`.

        :param neighbrohood: neighborhood property of a point cloud
        :param normals: normals property of a point cloud
        :param roughness: minimal object size - usually refers to surface roughness
        :param alpha: confidence value for statistical test
        :param min_points_in_neighborhood: minimal number of points in a neighborhood to make it viable for curvature computation. Default: 8
        :param min_points_in_sector: minimal points in a sector to be considered valid. Default: 1
        :param valid_sectors: minimal sectors needed for a point to be considered good. Default: 7
        :param num_sectors: the number of sectors the circle is divided to. Default: 8
        :param invalid_value: value for invalid curvature (points that their curvature cannot be computed). Default: -999
        :param curvatureProperty: A curvature property to update with the umbrella curvature. If None, a new curvature property will be created, with the principal curvatures as None.
        :param verbose: print inter running messages. default: False

        :type neighbrohood: NeighborsProperty.NeighborsProperty
        :type normals: NormalsProperty.NormalsProperty
        :type roughness: float
        :type alpha: float
        :type min_points_in_neighborhood: int
        :type min_points_in_sector: int
        :type valid_sectors: int
        :type num_sectors: int
        :type invalid_value: float
        :type curvatureProperty: CurvatureProperty
        :type verbose: bool

        :return: umbrella curvature for all points that have enough neighbors for the curvature to be computed.

        :rtype: np.array

        .. seealso::

           :meth:`__good_point` and :meth:`__checkNeighborhood`

        """
        from scipy import stats
        from Properties.Neighborhood.PointNeighborhood import PointNeighborhood

        epsilon = stats.norm.ppf(1 - alpha / 2) * roughness # Z-distribution
        umbrellaCurvature = []

        for point_neighbors in tqdm(neighbrohood, total=neighbrohood.Size,
                                    desc='Compute umbrella curvature for all point cloud', position=0):
            # check if the neighborhood is valid
            if isinstance(point_neighbors, PointNeighborhood) and cls.__checkNeighborhood(point_neighbors,
                                                                                       min_points_in_neighborhood=min_points_in_neighborhood,
                                                                                       min_points_in_sector=min_points_in_sector,
                                                                                       valid_sectors=valid_sectors,
                                                                                       num_sectors=num_sectors):
                point_idx = point_neighbors.neighborhoodIndices[0]
                if point_idx == 4230:
                    print('!')
                n = normals.Normals[point_idx]

                # if point_neighbors.center_point_coords[2] >= 5:
                #     from VisualizationO3D import VisualizationO3D
                #     vis = VisualizationO3D()
                #     vis.visualize_pointset(point_neighbors.neighbors)

                if verbose:
                    print(point_idx, n)

                # compute the direction projections  on the normal at the center point of each neighbor
                projections = point_neighbors.neighbors_vectors().dot(n) * point_neighbors.distances[1:]

                # check if the projections are statistically zero
                # projections[np.where(np.abs(projections) < epsilon)] = 0
                if np.abs((np.sum(projections) / point_neighbors.numberOfNeighbors)) < epsilon:
                    umbrellaCurvature.append(0)
                else:
                    umbrellaCurvature.append(np.sum(projections) / point_neighbors.numberOfNeighbors)
            else:
                if verbose:
                    print('invalid point:', point_neighbors.center_point_idx)
                umbrellaCurvature.append(invalid_value)
        umbrella_curvature = np.asarray(umbrellaCurvature)
        if verbose:
            print(umbrella_curvature.mean())
        if curvatureProperty is None:
            return CurvatureProperty(neighbrohood.Points, principal_curvatures=None,
                                     umbrella_curvature=umbrella_curvature)
        else:
            curvatureProperty.load(None, umbrella_curvature=umbrella_curvature)
            return curvatureProperty

    @classmethod
    def filter_curvature_roughness(cls, curvature_property, attribute_name, mean=0, std=1, alpha=0.05,
                                   verbose=False, posonly=False):
        r"""
        Turns curvature values to zero if they are part of a normal distribution N~(mean, std)

        .. math::
            H_0:\qquad \left|\kappa \right| \leq Z_{1-\frac{\alpha}{2}}

        :param curvature_property: the curvature property to which the filterization is applied
        :param attribute_name: the curvature attribute to filter
        :param mean: the mean value which will turn to zero if a value is part of the population. Default: 0
        :param std: the standard deviation from the mean value. Usually defined by the data texture/roughness. Defualt: 1
        :param alpha: the confidence level according to which a value is measured
        :param verbose: print inter-running messages

        :type curvature_property: CurvatureProperty
        :type attribute_name: str
        :type mean: float
        :type std: float
        :type alpha: float
        :type verbose: bool

        :return: a new curvature property with filterized values. All values that were in the old property are kept.

        :rtype: CurvatureProperty
        """
        from scipy import stats

        w_alpha = stats.norm.ppf(1 - alpha / 2)
        new_curvature = curvature_property
        curvature_values = curvature_property.__getattribute__(attribute_name)

        # check that the value statistically differ from the mean
        k = (curvature_values - mean) / std

        new_values = curvature_values.copy()
        zeros_ = np.abs(k) < w_alpha
        new_values[zeros_] = 0
        if posonly:
            new_values[new_values > 0] = 0

        new_curvature.__setattr__(attribute_name, new_values)
        return new_curvature

    @classmethod
    def curvature_numeric(cls, raster, ksize=3, sigma=2.5, gradientType='L1'):
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

        dx, dy, Zxx, Zyy, Zxy = mt.computeImageDerivatives_numeric(img, 2, sigma=sigma,
                                                           gradientType=gradientType,
                                                           ksize=ksize)

        k1 = (((Zxx + Zyy) + np.sqrt((Zxx - Zyy) ** 2 + 4 * Zxy ** 2)) / 2)
        k2 = (((Zxx + Zyy) - np.sqrt((Zxx - Zyy) ** 2 + 4 * Zxy ** 2)) / 2)

        if isinstance(raster, RasterData):
            return CurvatureProperty(raster, np.concatenate((k1[:, :, None], k2[:, :, None]), axis=2))
        else:
            return np.concatenate((k1[:, :, None], k2[:, :, None]), axis=2)

    @classmethod
    def checkNeighborhood(cls, neighborhood, min_points_in_neighborhood=5, min_points_in_sector=2, valid_sectors=7,
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
        return cls.__checkNeighborhood(neighborhood, min_points_in_neighborhood, min_points_in_sector, valid_sectors,
                                       num_sectors)

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
    def __BiQuadratic_Surface(neighbor_points, testCoeff=None, alpha=None):
        r'''
        BiQuadratic surface adjustment to discrete point cloud (when the surface is rotated to [0,0,1])

        :param neighbor_points: 3D points coordinates
        :param testCoeff: array of values to statistically test the results against. Default: no testing (None)
        :param alpha: for the F test

        :type neighbor_points: nx3 array
        :type testCoeff: np.ndarray
        :type alpha: float

        For testing the hypothesis that the unknowns are equal to testCoeff, i.e.,: :math:`H_0: {\bf x}=\{bf x_0}` we use the statistic:

        .. math::
            \frac{S_H}{u} = \frac{1}{u} \left({\bf x}-{\bf x_0}\right)^T\Sigma_x^{-1}\left({\bf x}-{\bf x_0}\right)

        and compare it to the F-value :math:`F_{u, n-u, \alpha}`, where: u the number of observations and n the number of unknowns. If :math:`\frac{S_H}{u} > F_{u, n-u, \alpha}
        then we reject the null hypothesis.

        :return: p - surface's coefficients, and Ftest result (True if rejected, False if was not rejected)

        :rtype: nd-array

        '''
        from numpy.linalg import LinAlgError

        Ftest_reject = False # do not reject

        x = np.expand_dims(neighbor_points[:, 0], 1)
        y = np.expand_dims(neighbor_points[:, 1], 1)
        z = np.expand_dims(neighbor_points[:, 2], 1)

        A = np.hstack((x ** 2, y ** 2, x * y))
        N = A.T.dot(A)
        u = A.T.dot(z)


        try:  # try to fit a bi-quadratic surface
            p = np.linalg.solve(N, u)

        except LinAlgError: # if the surface is a plane, fit a plane through homogeneous solution
            from Utils.MyTools import eig
            A = np.hstack((x, y ,z))
            N = A.T.dot(A)
            eigvals, eigvecs = eig(N)
            p = eigvecs[:, 0][:, None]
            p.reshape((3, 1))

        v = A.dot(p) - z
        sigma2 = v.T.dot(v) / (neighbor_points.shape[0] - 3)
        Sigma = sigma2 * np.linalg.inv(N) # TODO: check how this works with the eigenvalues solution

        if testCoeff is not None:
            from scipy.stats import f
            F_statistic = f.ppf(alpha, 3, neighbor_points.shape[0])
            statistic = 1/ neighbor_points.shape[0] * (p-testCoeff).T.dot(np.linalg.inv(Sigma)).dot(p-testCoeff)

            if statistic > F_statistic: # rejecting the null hypothesis
                Ftest_reject = True

            return p, Ftest_reject
        else:
            return p
