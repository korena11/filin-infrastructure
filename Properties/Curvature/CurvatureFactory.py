'''
|today|
'''

import functools

import numpy as np
from numpy import ndarray

import RotationUtils
from CurvatureProperty import CurvatureProperty
from EigenFactory import EigenFactory
from NeighborsFactory import NeighborsFactory
from PointSet import PointSet
from RasterData import RasterData


class CurvatureFactory:
    '''
    curvature parameters computation
    '''

    @classmethod
    def Curvature_FundamentalForm(cls, ind, points, search_radius, max_nn=20, tree=None, ):
        r'''
        Curvature computation based on fundamental form when fitting bi-quadratic surface.

        Main idea:

        1. Find neighbors

        2. Rotate so that the normal will be [0 0 1]
            the bi-quadratic surface coefficients :math:`d,e,f` are zero and the fitting is of three parameters:

            .. math::

                z(x,y) = ax^2 + by^2 + cxy

        3. Second derivatives dictate the curvature

           .. math::
               {\bf H} = \begin{bmatrix} 2a & c \\ c & 2b \end{bmatrix}

        :param points: the point cloud
        :param ind: index of the point where the curvature is computed
        :param points: pointset
        :param tree: KD tree, if exists
        :param search_radius: radius of the neighborhood
        :param max_nn: maximum neighbors to use, default: 20

        :type ind: int
        :type points: PointSet or PointSetOpen3D
        :type tree: KDtree
        :type search_radius: float
        :type max_nn: int

        .. note::

            If the point cloud is sent as a `PointSet`, there's no need to define the maximum number of points. The normals
            for each point are computed by PCA, when there are more than 5 neighbors.

            If the point cloud is sent as a `PointSetOpen3D`, maximum number of neighbors should be defined. And the normals
            are computed via PCA *within* by Open3D

        :return: principal curvatures :math:`\kappa_1` and :math:`\kappa_2`

        :rtype: np.ndarray

        '''
        from PointSetOpen3D import PointSetOpen3D
        # find point's neighbors in a radius
        if isinstance(points, PointSetOpen3D):
            neighbors_diff = NeighborsFactory.GetPointNeighborsByID(points, ind, search_radius, max_nn,
                                                                    useOriginal=False)
            neighbors = neighbors_diff.neighbors

            if neighbors_diff.numberOfNeighbors - 1 > 5:
                point_quality = CurvatureFactory.__good_point(neighbors)

            else:
                k1, k2 = -999, -999
                return np.array([k1, k2])

        else:
            neighbor, tree = NeighborsFactory.GetNeighborsIn3dRange_KDtree(ind, points, search_radius, tree)
            neighbors = neighbor.ToNumpy()
            pnt = points.GetPoint(ind)

            # if there are more than 5 neighbors normals can be computed via PCA, if less -- the curvature won't be
            # computed
            if neighbors[1::, :].shape[0] > 5:
                neighbors = (neighbors - np.repeat(np.expand_dims(pnt, 0), neighbors.shape[0], 0))[1::, :]
                eigVal, eigVec = EigenFactory.eigen_PCA(neighbors, search_radius)

                normP = eigVec[:, np.where(eigVal == np.min(eigVal))[0][0]]
                # if a normal of a neighborhood is in an opposite direction rotate it 180 degrees
                if np.linalg.norm(pnt, 2) < np.linalg.norm(pnt + normP, 2):
                    normP = -normP
                n = np.array([0, 0, 1])

                # rotate the neighborhood to xy plane
                rot_mat = RotationUtils.Rotation_2Vectors(normP, n)

                neighbors = (np.dot(rot_mat, neighbors.T)).T
                pnt = np.array([0, 0, 0])
                point_quality = CurvatureFactory.__good_point(neighbors)
            else:
                k1, k2 = -999, -999
                return np.array([k1, k2])

        if point_quality == 1:
            p = CurvatureFactory.__BiQuadratic_Surface(np.vstack((neighbors)))
            Zxx = 2 * p[0]
            Zyy = 2 * p[1]
            Zxy = p[2]

            k1 = (((Zxx + Zyy) + np.sqrt((Zxx - Zyy) ** 2 + 4 * Zxy ** 2)) / 2)[0]
            k2 = (((Zxx + Zyy) - np.sqrt((Zxx - Zyy) ** 2 + 4 * Zxy ** 2)) / 2)[0]
        else:
            k1, k2 = -999, -999

        return np.array([k1, k2])

    @classmethod
    def read_or_calculate_curvature_data(cls, curves_path, pointset3d, localNeighborhoodParameters,
                                         delete_non_computed=False,
                                         verbose=True):
        """
        Reads curvature data or computes it

         .. code-block:: python

            CurvatureFactory.read_or_calculate_curvature_data('/Curvature/', p3d, {'r': 0.015, 'nn': 5})

        :param curves_path: path to file. If exists, the curvature is loaded; else, it computes it
        :param pointset3d: the points as PointSetOpen3D
        :param localNeighborhoodParameters: search radius ('r') and maximum nearest neighbors ('nn')
        :param verbose: print interim messages
        :param delete_non_computed: flag to delete or leave points that their curvature wasn't calculated. If False, these curvature values are set to -999. Default: False

        :type curves_path: str
        :type pointset3d: PointSetOpen3D.PointSetOpen3D
        :type localNeighborhoodParameters: dict
        :type verbose: bool
        :type delete_non_computed: bool

        :return: computed curvatures as Curvature property
        :rtype: CurvatureProperty

        """
        import open3d as O3D
        search_radius = localNeighborhoodParameters['search_radius']
        max_nn = localNeighborhoodParameters['maxNN']

        try:
            open_file = open(curves_path)
            file_lines = open_file.read()
            open_file.close()

            # Splitting into list of lines
            lines = file_lines.split('\n')
            del file_lines

            # Removing the last line if it is empty
            num_lines = len(lines)
            while True:
                if lines[num_lines - 1] == "":
                    num_lines -= 1
                    lines = lines[0: num_lines]
                else:
                    break

            curves_data = list(map(lambda k1_k2: np.float32(k1_k2.split(' ')), lines))
            curves = CurvatureProperty(pointset3d, np.array(curves_data))
            try:
                normals_load = np.loadtxt(curves_path + 'normals_', delimiter=' ',
                                          converters={0: lambda s: float(s.strip()),
                                                      1: lambda s: float(s.strip()),
                                                      2: lambda s: float(s.strip())})
                pointset3d.pointsOpen3D.normals = O3D.Vector3dVector(normals_load)


                if verbose:
                    print(
                        ">>> Calculating points normals. Neighborhood Parameters - r:" + str(
                            search_radius) + "\t nn:" + str(
                            max_nn))
            except IOError:
                pointset3d.CalculateNormals(searchRadius=search_radius, maxNN=max_nn, verbose=verbose)

        except IOError:
            try:
                normals_load = np.loadtxt(curves_path + 'normals_', delimiter=' ',
                                          converters={0: lambda s: float(s.strip()),
                                                      1: lambda s: float(s.strip()),
                                                      2: lambda s: float(s.strip())})
                import open3d as O3D
                pointset3d.pointsOpen3D.normals = O3D.Vector3dVector(normals_load)
            except IOError:
                pointset3d.CalculateNormals(searchRadius=search_radius, maxNN=max_nn, verbose=verbose)
                np.savetxt(curves_path + 'normals_', np.asarray(pointset3d.pointsOpen3D.normals))

            if verbose:
                print(
                    ">>> Calculating points curvatures. Neighborhood Parameters - r:" + str(
                        search_radius) + "\t nn:" + str(
                        max_nn))

            curves_data = list(
                map(functools.partial(CurvatureFactory.Curvature_FundamentalForm, points=pointset3d,
                                      search_radius=search_radius,
                                      max_nn=max_nn), range(len(pointset3d.pointsOpen3D.points))))
            curves_data = np.array(curves_data)
            np.savetxt(curves_path, curves_data)
            curves = CurvatureProperty(pointset3d, curves_data)

        print("Number of curvatures computed before filterization: ", (curves.Size))
        if delete_non_computed:
            curves = CurvatureFactory.__keep_good_curves_data(curves.getValues(), pointset3d)
            print("Number of curvatures computed after filterization: ", curves.Size)

        return curves

    @classmethod
    def curvature_raster_fundamentalForm(cls, raster, ksize=3, sigma=2.5, gradientType='L1'):
        """
        Compute raster curvature based on first fundamental form

        :param raster: the raster image
        :param ksize: window/kernel size (equivalent to neighborhood size)
        :param sigma: for Gaussian blurring
        :param gradientType: 'L1' or 'L2' for derivatives computation

        :type raster: ndarray or RasterData
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
    @staticmethod
    def __good_point(neighbors):
        '''
        Determine whether the point is appropriate for curvature calculation

        1. Calculate Azimuth for all neighborhood points
        2. Check number of points in every 45 degree sector
        '''
        neighbor_points = neighbors.ToNumpy()
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

        for i in np.linspace(0, 7.0 * np.pi / 4.0, 8):
            p_in_sector = (np.where(p_angle[np.where(p_angle <= i + np.pi / 4.0)] > i))[0].size
            if p_in_sector >= 2:  # original: rad * 85:  # Why this threshold was chosen? I changed it to 2
                count_valid_sectors += 1

        if count_valid_sectors >= 7:
            return 1
        else:
            return 0

    @staticmethod
    def __keep_good_curves_data(curves_data, pointset_open3D):
        """

        :param curves_data: an array of curvature data
        :param pointset_open3D: the point set that relates to the curvature data

        :type curves_data: np.ndarray
        :type pointset_open3D: PointSetOpen3D

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
