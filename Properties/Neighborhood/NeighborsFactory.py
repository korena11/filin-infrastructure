import numpy as np
from matplotlib.path import Path
from numpy import mean, round, nonzero, where, hstack, inf, rad2deg, expand_dims
from scipy.spatial import kdtree as cKDTree
from tqdm import tqdm

from DataClasses.PointSubSet import PointSubSet, PointSet
from DataClasses.PointSubSetOpen3D import PointSetOpen3D, PointSubSetOpen3D
from Properties.Neighborhood.NeighborsProperty import NeighborsProperty
from Properties.Neighborhood.PointNeighborhood import PointNeighborhood
from Properties.Transformations.SphericalCoordinatesFactory import SphericalCoordinatesFactory


class NeighborsFactory:
    """
    Find neighbors of a given points using different methods. Use this factory to create either PointNeighbors or
    NeighborProperty for a whole point cloud.
    """

    @classmethod
    def buildNeighbors_rnn(cls, pointset, search_radius, method=None):
        """
        A generic method to build NeighborsProperty based on search radius (RNN)

        :param pointset: the cloud to which the NeighborhoodProperty should be computed
        :param search_radius: the neighborhood radius
        :param method: the method that should be used for computation

        :type pointset: PointSet.PointSet, BallTreePointSet.BallTreePointSet, PointSetOpen3D.PointSetOpen3D
        :type search_radius: float
        :type method: ufunc

        :return:  Neighbors Property for the whole cloud

        :rtype: NeighborsProperty

        **Usage Example**

        .. literalinclude:: ../../../../Properties/Neighborhood/test_neighborsFactory.py
            :lines: 87-90
            :emphasize-lines: 4
            :linenos:

        """
        from DataClasses.BallTreePointSet import BallTreePointSet
        from DataClasses.PointSetOpen3D import PointSetOpen3D
        from DataClasses.KdTreePointSet import KdTreePointSet
        from warnings import warn

        neighbors = -1

        if method is None:
            try:
                neighbors = NeighborsFactory.balltreePointSet_rnn(pointset, search_radius)

            except:
                warn('Could not compute with ball tree, trying open3d')
                neighbors = NeighborsFactory.pointSetOpen3D_rnn_kdTree(pointset, search_radius)
        else:
            try:
                neighbors = method(pointset, search_radius)
            except:
                if method.__name__ == 'balltreePointSet_rnn':
                    print('Build Ball Tree with default leaf size ')
                    pointset = BallTreePointSet(pointset)
                    neighbors = method(pointset, search_radius)

                elif method.__name__ == 'pointSetOpen3D_rnn_kdTree':
                    print('Build PointSetOpen3D object')
                    pointset = PointSetOpen3D(pointset)
                    neighbors = method(pointset, search_radius)

                elif method.__name__ == 'kdtreePointSet_rnn':
                    print('Build KDTreePointSet object')
                    pointset = KdTreePointSet(pointset)
                    neighbors = method(pointset, search_radius)

        if neighbors == -1:
            neighbors = -1
            warn(RuntimeError, 'Cannot compute neighbors ')
        return neighbors

    @classmethod
    def buildNeighbors_knn(cls, pointset, k_nearest_neighbors, method=None):
        """
        A generic method to build NeighborsProperty based on k-nearest-neighbors (KNN)

        :param pointset: the cloud to which the NeighborhoodProperty should be computed
        :param k_nearest_neighbors: the number of neighbors
        :param method: the method that should be used for computation

        :type pointset: PointSet.PointSet, BallTreePointSet.BallTreePointSet, PointSetOpen3D.PointSetOpen3D
        :type k_nearest_neighbors: int
        :type method: ufunc

        :return:  Neighbors Property for the whole cloud

        :rtype: NeighborsProperty

        **Usage Example**

        .. literalinclude:: ../../../../Properties/Neighborhood/test_neighborsFactory.py
            :lines: 93-106
            :emphasize-lines: 11
            :linenos:

        """
        from DataClasses.BallTreePointSet import BallTreePointSet
        from DataClasses.PointSetOpen3D import PointSetOpen3D
        from DataClasses.KdTreePointSet import KdTreePointSet
        from warnings import warn

        neighbors = -1

        if method is None:
            try:
                neighbors = NeighborsFactory.balltreePointSet_knn(pointset, k_nearest_neighbors)

            except:
                warn('Could not compute with ball tree, trying open3d')
                neighbors = NeighborsFactory.pointSetOpen3D_rnn_kdTree(pointset, k_nearest_neighbors)

        else:
            try:
                neighbors = method(pointset, k_nearest_neighbors)
            except:
                if method.__name__ == 'balltreePointSet_knn':
                    print('Build Ball Tree with default leaf size ')
                    pointset = BallTreePointSet(pointset)
                    neighbors = method(pointset, k_nearest_neighbors)

                elif method.__name__ == 'pointSetOpen3D_knn_kdTree':
                    print('Build PointSetOpen3D object')
                    pointset = PointSetOpen3D(pointset)
                    neighbors = method(pointset, k_nearest_neighbors)

                elif method.__name__ == 'kdtreePointSet_rnn':
                    print('Build KDTreePointSet object')
                    pointset = KdTreePointSet(pointset)
                    neighbors = method(pointset, k_nearest_neighbors)

        if neighbors == -1:
            neighbors = -1
            warn(RuntimeError, 'Cannot compute neighbors ')
        return neighbors

    @staticmethod
    def balltreePointSet_rnn(pointset_bt, search_radius, verbose=False):
        """
        Create NeighborsProperty of BallTreePointSet (whole cloud) based on search radius (RNN)

        :param pointset_bt: the cloud to which the NeighborhoodProperty should be computed
        :param search_radius: the neighborhood radius
        :param verbose: print running messages

        :type pointset_bt: BallTreePointSet.BallTreePointSet
        :type search_radius: float
        :type verbose: bool

        :return: NeighborsProperty
        """

        print('>>> Find all points neighbors using Ball Tree')

        neighbors = NeighborsProperty(pointset_bt)  # initialization of the neighborhood property

        idx = pointset_bt.queryRadius(pointset_bt.ToNumpy(), search_radius)

        for id in tqdm(range(len(idx)), total=len(idx), desc='rnn neighbors by ball tree', position=0):
            current_id = np.asarray(idx[id])
            tmp_subset = PointSubSet(pointset_bt, current_id)
            tmp_point_neighborhood = PointNeighborhood(tmp_subset)
            neighbors.setNeighborhood(id, tmp_point_neighborhood)

        return neighbors

    @staticmethod
    def balltreePointSet_knn(pointset_bt, k_nearest_neighbors):
        """
        Create NeighborsProperty of BallTreePointSet (whole cloud) based on k-nearest-neighbors (RNN)

        :param pointset_bt: the cloud to which the NeighborhoodProperty should be computed
        :param k_nearest_neighbors: the number of neighbors

        :type pointset_bt: BallTreePointSet.BallTreePointSet
        :type k_nearest_neighbors: int

        :return: NeighborsProperty

        .. seealso::
           :meth:`balltreePointSet_rnn`

        """
        print('>>> Find all points neighbors using ball tree')

        neighbors = NeighborsProperty(pointset_bt)  # initialization of the neighborhood property

        idx = pointset_bt.query(pointset_bt.ToNumpy(), k_nearest_neighbors)

        for id in tqdm(range(len(idx)), total=len(idx), desc='knn neighbors by ball tree', position=0):
            current_id = np.asarray(idx[id])
            tmp_subset = PointSubSet(pointset_bt, current_id)
            tmp_point_neighborhood = PointNeighborhood(tmp_subset)
            neighbors.setNeighborhood(id, tmp_point_neighborhood)

        return neighbors

    @staticmethod
    def kdtreePointSet_rnn(pointset_kdt, search_radius, k_nearest_neighbors=None, parts_size=int(5e5), parts_num=None, **kwargs):
        r"""
        Create NeighborsProperty of KdTreePointSet (whole cloud) based on search radius (RNN)

        :param pointset_kdt: the cloud to which the NeighborhoodProperty should be computed
        :param search_radius: the neighborhood radius
        :param k_nearest_neighbors: maximum number or neighbors. If sent, only the farthest k-neighbors will be stored. When None, there will be no limit. Default: None.
        :param parts_size: number of points in section for more efficient computation. Default: None
        :param parts_num: number of parts to divide the computation. Default: 1

        :type pointset_kdt: KdTreePointSet.KdTreePointSet
        :type search_radius: float
        :type k_nearest_neighbors: int
        :type parts_size: int
        :type parts_num: int

        :return: NeighborsProperty
        :rtype: NeighborsProperty

        .. warning::
            Division to parts doesn't work

        .. warning::
            The kmax doesn't work. It doesn't put the first point as the center point.
        """
        # TODO: division to parts doesn't work. Need to solve the mapping at the end

        from tqdm import trange
        print('>>> Find all points neighbors in radius %f using kd-Tree' % search_radius)

        neighbors = NeighborsProperty(pointset_kdt)  # initialization of the neighborhood property
        if parts_num is None and parts_size is None:
            parts_num = 1
            modulu = 0
        elif parts_num is None:
            parts_num = int(pointset_kdt.Size / parts_size)
            modulu = pointset_kdt.Size % parts_size
        else:
            modulu = pointset_kdt.Size % parts_num
            parts_size = int(pointset_kdt.Size / parts_num)

        start = 0
        idx = []
        for part in trange(parts_num):
            start = int(part * parts_size)

            if part == 0:
                if parts_num == 1:  # patch because the parts dont work. Delete when fixed
                    idx = (pointset_kdt.queryRadius(pointset_kdt.ToNumpy()[start:start + parts_size], search_radius,
                                                    sort_results=True))

                else:
                    idx = np.hstack((idx,
                                     pointset_kdt.queryRadius(pointset_kdt.ToNumpy()[start:start + parts_size],
                                                              search_radius,
                                                              sort_results=True)))
            else:
                idx = np.hstack((idx,
                                 pointset_kdt.queryRadius(pointset_kdt.ToNumpy()[start:start + parts_size],
                                                          search_radius,
                                                          sort_results=True)))
        # for the remaining part
        # if parts_num != 1:
        if modulu > 0:
            idx = np.hstack((idx,
                             pointset_kdt.queryRadius(pointset_kdt.ToNumpy()[parts_size * parts_num:], search_radius,
                                                      sort_results=True)))
        if k_nearest_neighbors is not None:
            start = k_nearest_neighbors
        else:
            start = -0

        pointSubSets = list(map(lambda id: PointSubSet(pointset_kdt, id[-start:]), tqdm(idx)))
        pointNeighborhoods = list(map(lambda pntSubSet: PointNeighborhood(pntSubSet), tqdm(pointSubSets)))
        list(map(neighbors.setNeighborhood, range(pointset_kdt.Size), tqdm(pointNeighborhoods)))

        return neighbors

    @staticmethod
    def kdtreePointSet_knn(pointset_kdt, k_nearest_neighbors, parts_size=int(5e5), parts_num=None):
        """
        Create NeighborsProperty of KdTreePointSet (whole cloud) based on k-nearest-neighbors (KNN)

        :param pointset_kdt: the cloud to which the NeighborhoodProperty should be computed
        :param k_nearest_neighbors: the number of neighbors
        :param parts_size: number of points in section for more efficient computation. Defauls: 5e5
        :param parts_num: number of parts to divide the computation. Defualt: None

        :type pointset_kdt: KdTreePointSet.KdTreePointSet
        :type k_nearest_neighbors: int
        :type parts_size: int
        :type parts_num: int

        :return: NeighborsProperty

        .. seealso::
           :meth:`kdtreePointSet_rnn`

        """
        from tqdm import trange
        print('>>> Find all {k} points neighbors using kd-tree'.format(k=k_nearest_neighbors))

        neighbors = NeighborsProperty(pointset_kdt)  # initialization of the neighborhood property
        if parts_size > pointset_kdt.Size:
            parts_num = 1
            parts_size = pointset_kdt.Size
            modulu = 0
        else:
            if parts_num is None:
                parts_num = int(pointset_kdt.Size / parts_size)
                modulu = pointset_kdt.Size % parts_size
            else:
                modulu = pointset_kdt.Size % parts_num
                parts_size = int(pointset_kdt.Size / parts_num)

        start = 0
        idx = None

        for part in trange(parts_num, position=0):
            start = int(part * parts_size)

            if part == 0:
                idx = pointset_kdt.query(pointset_kdt.ToNumpy()[start:start + parts_size], k_nearest_neighbors)
            else:
                idx = np.vstack(
                    (idx, pointset_kdt.query(pointset_kdt.ToNumpy()[start:start + parts_size], k_nearest_neighbors)))

        # for the remaining part
        if modulu > 0:
            idx = np.vstack((idx, pointset_kdt.query(pointset_kdt.ToNumpy()[start + parts_size:], k_nearest_neighbors)))

        pointSubSets = list(map(lambda id: PointSubSet(pointset_kdt, id), idx))
        pointNeighborhoods = list(map(lambda pntSubSet: PointNeighborhood(pntSubSet), pointSubSets))
        list(map(neighbors.setNeighborhood, range(pointset_kdt.Size), pointNeighborhoods))

        return neighbors

    @staticmethod
    def pointSetOpen3D_rnn_kdTree(pointset3d, search_radius, verbose=False):
        """
        Create NeighborsProperty of PointSetOpen3D (whole cloud) based on search radius (RNN)

        :param pointset3d:  the cloud to which the NeighborProperty should be computed
        :param search_radius: the neighborhood radius

        **Optionals**

        :param verbose: print inter-running

        :type pointset3d: PointSetOpen3D.PointSetOpen3D
        :type search_radius: float
        :type verbose: bool

        :return: a property consisting of the PointNeighborhood for each point in the cloud

        :rtype: NeighborsProperty

        .. seealso::

            `FLANN <https://www.cs.ubc.ca/research/flann/>`_ , :meth:`pointSetOpen3D_knn_kdTree`, :meth:`point3d_neighbors_kdtree`, :meth:`pointSetOpen3D_rknn_kdTree`

        """
        from DataClasses.PointSubSetOpen3D import PointSubSetOpen3D
        print('>>> Find all points neighbors using open3d')

        neighbors = NeighborsProperty(pointset3d)

        for point, i in tqdm(zip(pointset3d, range(pointset3d.Size)), total=pointset3d.Size,
                             desc='Compute rnn neighbors for all point cloud', position=0):
            k, idx, distances = pointset3d.kdTreeOpen3D.search_radius_vector_3d(point,
                                                                                search_radius)
            distances = np.asarray(distances)
            idx = np.asarray(idx)

            # create a temporary neighborhood
            tmp_subset = PointSubSetOpen3D(pointset3d, idx)
            tmp_point_neighborhood = PointNeighborhood(tmp_subset, distances)
            neighbors.setNeighborhood(i, tmp_point_neighborhood)
        print('size {}, rad {}'.format(neighbors.average_neighborhood_size(), neighbors.average_neighborhood_radius()))
        return neighbors

    @staticmethod
    def pointSetOpen3D_knn_kdTree(pointset3d, k_nearest_neighbors):
        """
        Create NeighborsProperty of PointSetOpen3D (whole cloud) based on k-nearest neighbors (KNN)

        :param pointset3d:  the cloud to which the NeighborProperty should be computed
        :param k_nearest_neighbors: number of neighbors to search

        :type pointset3d: PointSetOpen3D.PointSetOpen3D
        :type k_nearest_neighbors: int

        :return: a property consisting of the PointNeighborhood for each point in the cloud

        :rtype: NeighborsProperty

        .. seealso::

            `FLANN <https://www.cs.ubc.ca/research/flann/>`_, :meth:`pointSetOpen3D_rnn_kdTree`, :meth:`point3d_neighbors_kdtree`, :meth:`pointSetOpen3D_rknn_kdTree`

        """
        from DataClasses.PointSubSetOpen3D import PointSubSetOpen3D
        print('>>> Find all points neighbors using open3d')

        neighbors = NeighborsProperty(pointset3d)

        for point, i in tqdm(zip(pointset3d, range(pointset3d.Size)), total=pointset3d.Size,
                             desc='Compute knn neighbors for all point cloud', position=0):
            k, idx, distances = pointset3d.kdTreeOpen3D.search_knn_vector_3d(point, k_nearest_neighbors + 1)
            distances = np.asarray(distances)
            if np.all(np.round(distances) == 0):
                distances = None
            idx = np.asarray(idx)

            # create a temporary neighborhood
            tmp_subset = PointSubSetOpen3D(pointset3d, idx)
            neighbors.setNeighborhood(i, PointNeighborhood(tmp_subset, distances))

        return neighbors

    @staticmethod
    def pointSetOpen3D_rknn_kdTree(pointset3d, k_nearest_neighbors, max_radius):
        """
        Create NeighborsProperty of PointSetOpen3D (whole cloud) using KRNN.

        Return a neighborhood with at most k nearest neighbors that have distances to the anchor point less than a given radius.

        :param pointset3d:  the cloud to which the NeighborProperty should be computed
        :param k_nearest_neighbors: number of neighbors to search
        :param max_radius: maximal radius of neighbors

        :type pointset3d: PointSetOpen3D.PointSetOpen3D
        :type k_nearest_neighbors: int
        :type max_radius: float

        :return: a property consisting of the PointNeighborhood for each point in the cloud

        :rtype: NeighborsProperty

        .. seealso::

            `FLANN <https://www.cs.ubc.ca/research/flann/>`_, :meth:`pointSetOpen3D_rnn_kdTree`, :meth:`pointSetOpen3D_knn_kdTree`, :meth:`point3d_neighbors_kdtree`

        """
        from DataClasses.PointSubSetOpen3D import PointSubSetOpen3D
        print('>>> Find all points neighbors using open3d')

        neighbors = NeighborsProperty(pointset3d)

        for point, i in tqdm(zip(pointset3d, range(pointset3d.Size)), total=pointset3d.Size,
                             desc='Compute rknn neighbors for all point cloud', position=0):
            k, idx, distances = pointset3d.kdTreeOpen3D.search_hybrid_vector_3d(point, radius=max_radius,
                                                                                max_nn=k_nearest_neighbors + 1)

            distances = np.asarray(distances)
            idx = np.asarray(idx)

            # create a temporary neighborhood
            tmp_subset = PointSubSetOpen3D(pointset3d, idx)
            tmp_point_neighborhood = PointNeighborhood(tmp_subset, distances)

            # set in neighbors property
            neighbors.setNeighborhood(i, tmp_point_neighborhood)

        return neighbors

    @staticmethod
    def point3d_neighbors_kdtree(pt, pointset3d, radius=None, knn=None):
        """
        Find neighbors to a single point based kd-tree.

        This method can be used either for RNN, KNN or RKNN, depends on the parameters sent. Used only with PointSetOpen3D

        :param pt: the point to look for (index or 3x1)
        :param pointset3d: the point cloud in which the neighbors are serached in
        :param radius: maximal radius for neighbors
        :param knn: number of neighbors to look for

        :type pt: np.array, int
        :type pointset3d: PointSetOpen3D.PointSetOpen3D
        :type radius: float
        :type knn: int

        .. note::

           If both radius and knn are sent, then the RNN is used, which means: it returns at most k nearest neighbors that have distances to the anchor point less than a given radius

        :return: point neighborhood

        :rtype: PointNeighborhood

        .. seealso::

            `FLANN <https://www.cs.ubc.ca/research/flann/>`_, :meth:`pointSetOpen3D_rnn_kdTree`, :meth:`pointSetOpen3D_knn_kdTree`, :meth:`pointSetOpen3D_rknn_kdTree`
        """
        from DataClasses.PointSubSetOpen3D import PointSubSetOpen3D

        # check which parameters were received
        if radius is not None:
            radius_flag = True
        else:
            radius_flag = False

        if knn is not None:
            knn_flag = True
        else:
            knn_flag = False

        # decide which method to use according to the parameters received
        if knn_flag and radius_flag:
            # check if the point is an array or an index
            if isinstance(pt, np.ndarray):
                k, idx, dist = pointset3d.kdTreeOpen3D.search_hybrid_vector_3d(pt, radius=radius, knn=knn + 1)
            else:
                k, idx, dist = pointset3d.kdTreeOpen3D.search_hybrid_vector_3d(pointset3d.data.points[pt],
                                                                               radius=radius, knn=knn + 1)

        elif knn_flag:
            # check if the point is an array or an index
            if isinstance(pt, np.ndarray):
                k, idx, dist = pointset3d.kdTreeOpen3D.search_knn_vector_3d(pt, knn + 1)
            else:
                k, idx, dist = pointset3d.kdTreeOpen3D.search_knn_vector_3d(pointset3d.data.points[pt], knn + 1)

        elif radius_flag:
            # check if the point is an array or an index
            if isinstance(pt, np.ndarray):
                k, idx, dist = pointset3d.kdTreeOpen3D.search_radius_vector_3d(pt, radius)
            else:
                k, idx, dist = pointset3d.kdTreeOpen3D.search_radius_vector_3d(pointset3d.data.points[pt], radius)
        else:
            raise IOError

        point_subset = PointSubSetOpen3D(pointset3d, idx)
        return PointNeighborhood(point_subset, dist)

    @staticmethod
    def GetNeighborsIn3dRange(points, x, y, z, radius):
        """                            
        Find all tokens (points) in radius range

        Find all points in range of the ball field with radius 'radius'.

        :param points: - PointSet
        :param x y z:  search point coordinates
        :param radius: Radius of ball field

        :type points: PointSet

        :return: points in ranges

        :rtype: PointSubSet

        """
        if points == None or points.Size == 0:
            return None

        if z == None:
            return None

        xs = points.X
        ys = points.Y
        zs = points.Z

        dists = []

        xm = mean(xs)
        x_ = round((xs - xm) / radius)
        ym = mean(ys)
        y_ = round((ys - ym) / radius)
        zm = mean(zs)
        z_ = round((zs - zm) / radius)

        x = round((x - xm) / radius)
        y = round((y - ym) / radius)
        z = round((z - zm) / radius)

        indices = nonzero((abs(x_ - x) <= 1) & (abs(y_ - y) <= 1) & (abs(z_ - z) <= 1))[0]

        for j in range(0, len(indices)):

            k = indices[j]
            dd = (x_[k] - x) ** 2 + (y_[k] - y) ** 2 + (z_[k] - z) ** 2

            if dd > 1:
                indices[j] = -1
            else:
                dists.append(dd * radius)

        indices = indices[indices > 0]

        pointsInRange = PointNeighborhood(PointSubSet(points, indices), dists)

        return pointsInRange

    @staticmethod
    def GetNeighborsIn3dRange_KDtree(ind, pntSet, radius, tree=None, num_neighbor=None):
        '''
        Find neighbors of a point using KDtree

        :param ind: search point coordinates
        :param pntSet: the points in which the neighbors are required - in cartesian coordinates
        :param radius: search radius
        :param tree: KD tree
        :param num_neighbor: number of nearest neighbors
                if num_neighbor!=None the result will be the exact number of neighbors 
                and not neighbors in radius
         
        :type ind: int
        :type pntSet: PointSet.PointSet
        :type tree: cKDTree
        :type num_neighbor: int

        :return: subset of neighbors from original pointset

        :rtype: PointSubSet

        '''

        if num_neighbor == None:
            pSize = pntSet.Size
        else:
            pSize = num_neighbor

        if tree == None:
            tree = cKDTree(pntSet.ToNumpy())
        pnt = pntSet.GetPoint(ind)
        l = tree.query(pnt, pSize, p=2, distance_upper_bound=radius)
        #         neighbor = PointSubSet(pntSet, l[1][where(l[0] != inf)[0]])
        neighbor = l[1][where(l[0] != inf)[0]]
        return PointSubSet(pntSet, neighbor), tree

    @staticmethod
    def GetNeighborsIn3dRange_SphericCoord(pnt, points, radius):
        '''
        Find points in defined window - neighbors of a point
        
        :param pnt: search point coordinates
        :param points: pointset - in spherical coordinates
        :param radius: search radius in radian

        :type pnt: tuple?
        :type points: PointSet.PointSet
        :type radius: float

        :return: neighbors from original pointset

        :rtype: PointSubSet

        '''
        radius = rad2deg(radius)
        az_min, az_max, el_min, el_max = pnt[0] - radius, pnt[0] + radius, pnt[1] - radius, pnt[1] + radius
        wind = Path([(az_min, el_max), (az_max, el_max), (az_max, el_min), (az_min, el_min)])
        wind.iter_segments()
        pntInCell = wind.contains_points(
            hstack((expand_dims(points.azimuths, 1), expand_dims(points.ElevationAngles, 1))))

        indices = nonzero(pntInCell)
        i1 = where(points.ranges[indices[0]] >= pnt[2] - 0.10)
        i2 = where(points.ranges[indices[0][i1[0]]] <= pnt[2] + 0.10)
        neighbors = SphericalCoordinatesFactory.CartesianToSphericalCoordinates(
            PointSubSet(points.XYZ, indices[0][i1[0][i2[0]]]))
        return neighbors


    @staticmethod
    def buildNeighbors_panorama(panorama, radius):
        """
        Build neighborhood property using panorama data structure and a radius (meters)

        :param panorama: a point cloud as a panorama
        :param radius: the physical world radius to search for neighbors (meters)

        :type panorama: PanoramaProperty.PanoramaProperty
        :type radius: float

        :return: Neighbors property

        :rtype: NeighborsProperty
        """
        from Properties.Panoramas.PanoramaUtils import boundPanoramaExpand
        # construct window sizes according to range
        # 1. window size at each cell
        d_az = np.ceil(radius/ (panorama.rangeImage * np.radians(panorama.azimuth_spacing))).astype('int')
        d_el = np.ceil(radius/ (panorama.rangeImage * np.radians(panorama.elevation_spacing))).astype('int')

        # 2. pad panorama according to cell size
        m, n = panorama.Size
        cols_indx = np.arange(0, n)
        rows_indx = np.arange(0, m)
        nn, mm = np.meshgrid(cols_indx, rows_indx)  # indices of the panorama

        column_L = np.abs((nn - d_az).min())  # leftmost column
        column_R = (nn + d_az).max() - n + 1 # rightmost column
        row_T = np.abs((mm - d_el).min())  # topmost row
        row_B = (mm + d_el).max() - m + 1  # lowwermost row
        nn += column_L
        mm += row_T

        img_extended = boundPanoramaExpand(panorama.panoramaIndex, row_T, column_L, row_B, column_R)
        neighborhood = NeighborsProperty(panorama.Points)

        for win_col in tqdm(np.unique(d_az), desc='finding neighbors', position=0, leave=False):
            for win_row in tqdm(np.unique(d_el), position=1, leave=False):
                indx = np.where(d_el == win_row) and np.where(d_az == win_col)

                for id in zip(indx[0], indx[1]):
                    if img_extended[id[0]+row_T, id[1] + column_L ] == -1:
                        continue

                    # neighbors_indx = np.array([img_extended[(row_T + id[0]) - win_row, (column_L + id[1]) - win_col],
                    #                   img_extended[(row_T + id[0]) - win_row, (column_L + id[1] )],
                    #                   img_extended[(row_T + id[0]) - win_row, (column_L + id[1]) + win_col],
                    #                   img_extended[(row_T + id[0]), (column_L + id[1]) - win_col],
                    #                   img_extended[(row_T + id[0]), (column_L + id[1])],
                    #                   img_extended[(row_T + id[0]), (column_L + id[1]) + win_col ],
                    #                   img_extended[(row_T + id[0]) + win_row, (column_L + id[1]) - win_col],
                    #                   img_extended[(row_T + id[0]) + win_row, (column_L + id[1])],
                    #                   img_extended[(row_T + id[0]) + win_row, (column_L + id[1]) + win_col]
                    #                   ])

                    neighbors_indx = img_extended[(row_T + id[0]) - win_row: (row_T + id[0])+ win_row + 1,
                                     (column_L + id[1]) - win_col:(column_L + id[1]) + win_col + 1]
                    n_idx = np.unique(neighbors_indx[neighbors_indx != -1])

                    pointsub = PointSubSet(panorama.Points, np.hstack((panorama.panoramaIndex[id],n_idx)))
                    neighborhood.setNeighborhood(panorama.panoramaIndex[id], PointNeighborhood( pointsub))
                    # print(neighborhood.getNeighborhood(panorama.panoramaIndex[id]).distances)

        return neighborhood
    @staticmethod
    def ComputeNeighbors_raster(points, res, radius):
        """
        Find neighbors for all dataset in raster or gridded point cloud

        :TODO: complete for RasterData.

        :param points: raster or gridded point cloud
        :param res: grid cell size
        :param radius: half window size (should be an odd number)

        :type points: DataClasses.BaseData.BaseData, DataClasses.PointSet.PointSet, DataClasses.PointSetOpen3D.PointSetOpen3D
        :type radius: int

        :return: neighborhood property

        :rtype: Properties.NeighborhoodProperty.NeighborhoodProperty

        ..warning::
           - RasterData are not implemented

        """
        import warnings
        if radius %2 == 0:
            warnings.warn('radius is an even number, window will be smaller than expected')

        pts_numpy = points.ToNumpy()
        ind_sorted = np.argsort(pts_numpy.view((str(pts_numpy.dtype) + ',' + str(pts_numpy.dtype) + ',' + str( pts_numpy.dtype))),
                                order=['f0', 'f1'], axis=0).view('int')

        num_cols = int((points.X.max() - points.X.min()) / res + 1)
        num_rows = int((points.Y.max() - points.Y.min()) / res + 1)

        ind_raster = ind_sorted[:,0].reshape((num_rows, num_cols))
        pcl_neighborhood = NeighborsProperty(points)
        print('>> window size is {}'.format(int((radius))))
        for i in tqdm(np.arange(0, num_cols), desc='Raster-based neighbors '):
            for j in np.arange(0, num_rows):
                # define the window
                start_i = int(i-(radius-1)/2)
                start_j = int(j-(radius-1)/2)
                end_i = int(i+(radius-1)/2 + 1)
                end_j = int(j+(radius-1)/2 + 1)

                if start_i < 0:
                    start_i = 0
                if start_j <0:
                    start_j = 0
                if end_i > num_rows:
                    end_i = num_rows
                if end_j > num_cols:
                    end_j = num_cols

                # find the points
                current_pts = np.hstack((ind_raster[i,j], ind_raster[start_i:end_i, start_j:end_j].flatten()))
                if isinstance(points, PointSetOpen3D):
                    neighbors = PointSubSetOpen3D(points, current_pts)
                else:
                    neighbors = PointSubSet(points, current_pts)
                point_neighborhood = PointNeighborhood(neighbors)
                pcl_neighborhood.setNeighborhood(ind_raster[i,j], point_neighborhood)
                # pcl_neighborhood.average_neighborhood_radius()

        return pcl_neighborhood

    @classmethod
    def load_or_create_NeighborhoodFile(cls, pts, folder, filename, neighborhood_function,
                                        search_radius=None, k_nearest_neighbors=None, parts_num=None, parts_size=int(5e5), overwrite=False):
        """
        Loads a neighborhood file if exists, otherwise, creates the neighborhood and saves it as pickle

        :param pts: point cloud
        :param folder: path for the  neighborhood file
        :param filename:  name of the  neighborhood file
        :param neighborhood_function: the function by which the neighborhood will be computed. Default :func:`cls.kdtreePointSet_rnn`
        :param k_nearest_neighbors: number of nearest neighbors
        :param search_radius:  radius of search
        :param parts_size: number of points in section for more efficient computation. Defauls: 5e5
        :param parts_num: number of parts to divide the computation. Default: None
        :param overwrite: recompute the neighborhood file, even if exists

        :type pts: DataClasses.BaseData.BaseData
        :type folder: str
        :type filename: str
        :type neighborhood_function: ufunc
        :type k_nearest_neighbors: int
        :type search_radius: float
        :type parts_num: int
        :type parts_size: int
        :type overwrite: bool

        :return: neighborhood property

        :rtype: Properties.Neighborhood.NeighborhoodProperty.NeighborhoodProperty

        .. TODO::
            The saving should be changed to save only the path of the point cloud and the neighbors for each point - not the property as it saves now as this one falls when the dataset is big
        """
        import os
        import pickle
        from os import path

        kwargs = dict(search_radius=search_radius, k_nearest_neighbors=k_nearest_neighbors, parts_num=parts_num, parts_size=parts_size)
        create_flag = overwrite
        # check if the folder exists. if not, create it
        if not path.isdir(folder):
            os.makedirs(folder)
            create_flag = True
        else:
            if path.exists(folder+filename +'.p'):
                if not create_flag:
                    neighborhood = pickle.load(open(folder + filename + '.p', 'rb'))
                    print(filename + 'neighborhood loaded')
            else:
                create_flag = True

        if create_flag:
            neighborhood = neighborhood_function(pts, **kwargs)
            pickle.dump(neighborhood, open(folder + filename + '.p', 'wb'))

        return neighborhood
