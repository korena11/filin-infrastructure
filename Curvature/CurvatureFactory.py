import numpy as np

import RotationUtils
from EigenFactory import EigenFactory
from NeighborsFactory import NeighborsFactory
from PointSet import PointSet
from PointSetOpen3D import PointSetOpen3D


class CurvatureFactory:
    '''
    curvature parameters computation
    '''

    @staticmethod
    def Curvature_FundamentalForm(ind, points, search_radius, max_nn=None, tree=None, ):
        '''
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
        :param max_nn: maximum neighbors to use

        :type ind: int
        :type points: PointSet or PointSetOpen3D
        :type tree: KDtree
        :type search_radius: float
        :type max_nn: int

        .. note::

            If the point cloud is sent as a PointSet, there's no need to define the maximum number of points. The normal
            at the point is computed by PCA if there are more than 5 neighbors.

            If the point cloud is sent as a PointSetOpen3D, maximum number of neighbors should be defined. And the normals
            are computed via PCA *within* by Open3D

        :return: principal curvatures k1 and k2

        :rtype: np.ndarray

        '''
        # find point's neighbors in a radius
        if isinstance(points, PointSetOpen3D):
            neighbors_diff = points.GetPointsNeighborsByID(ind, search_radius, max_nn, useOriginal=False)
            localRotatedNeighborhood = neighbors_diff.localRotatedNeighborhood

            if neighbors_diff.numberOfNeighbors - 1 > 5:
                point_quality = CurvatureFactory.__good_point(localRotatedNeighborhood)

            else:
                k1, k2 = -999, -999
                return np.ndarray([k1, k2])

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

                localRotatedNeighborhood = (np.dot(rot_mat, neighbors.T)).T
                pnt = np.array([0, 0, 0])
                point_quality = CurvatureFactory.__good_point(neighbors)
            else:
                k1, k2 = -999, -999
                return np.ndarray([k1, k2])

        if point_quality == 1:
            p = CurvatureFactory.__BiQuadratic_Surface(np.vstack((localRotatedNeighborhood)))
            Zxx = 2 * p[0]
            Zyy = 2 * p[1]
            Zxy = p[2]

            k1 = (((Zxx + Zyy) + np.sqrt((Zxx - Zyy) ** 2 + 4 * Zxy ** 2)) / 2)[0]
            k2 = (((Zxx + Zyy) - np.sqrt((Zxx - Zyy) ** 2 + 4 * Zxy ** 2)) / 2)[0]
        else:
            k1, k2 = -999, -999

        return np.array([k1, k2])

    @staticmethod
    def __good_point(neighbor_points):
        '''
        Determine whether the point is appropriate for curvature calculation

        1. Calculate Azimuth for all neighborhood points
        2. Check number of points in every 45 degree sector
        '''

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
