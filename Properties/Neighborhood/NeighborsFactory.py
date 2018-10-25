import numpy as np
from matplotlib.path import Path
from numpy import mean, round, nonzero, where, hstack, inf, rad2deg, expand_dims
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

from NeighborProperty import NeighborsProperty
from PointNeighborhood import PointNeighborhood
# Infrastructure imports
from PointSet import PointSet
from PointSetOpen3D import PointSetOpen3D
from PointSubSet import PointSubSet
from PointSubSetOpen3D import PointSubSetOpen3D
from SphericalCoordinatesFactory import SphericalCoordinatesFactory


class NeighborsFactory:
    """
    Find neighbors of a given points using different methods. Use this factory to create either PointNeighbors or
    NeighborProperty for a whole point cloud.
    """

    @staticmethod
    def GetPointNeighborsByID(pointset3d, idx, searchRadius, maxNN, returnValues=True, neighborsProperty=None,
                              override=False, useOriginal=False, rotate=False):
        """
        Get the neighbors of a point within a point cloud by its index.

        :param pointset3d: the cloud in which the point is searched for
        :param idx: point index
        :param searchRadius: the search radius for neighbors
        :param maxNN: maximum number of neighbors
        :param returnValues: default: True
        :param neighborsProperty: Existing neighborhood property which should be override
        :param override: default: False
        :param rotate: flag whether to rotate the neighborhood or not. Default: False
        :param useOriginal: default: False

        :type pointset3d: PointSetOpen3D
        :type idx: int or np.ndarray
        :type searchRadius: float
        :type maxNN: int
        :type returnValues: bool
        :type neighborsProperty: NeighborsProperty
        :type override: bool
        :type useOriginal: bool
        :type rotate: bool

        :return: the point neighborhood

        :rtype: PointNeighborhood
        """
        if neighborsProperty is None:
            neighborsProperty = NeighborsProperty(pointset3d)

        pointNeighborhoodObject = 1
        if isinstance(idx, int):
            idx = [idx]

        if override:
            NeighborsFactory.__PrintOverrideNeighborhoodCalculations(neighborsProperty, idx[0], searchRadius, maxNN)

        for currentPointIndex in idx:
            if not override:
                if neighborsProperty.getNeighbors(currentPointIndex):
                    r = neighborsProperty.getNeighbors(currentPointIndex).GetRadius()
                    nn = neighborsProperty.getNeighbors(currentPointIndex).GetMaxNN()
                    if r == searchRadius and nn == maxNN:
                        continue

            currentPoint = pointset3d.GetPoint(currentPointIndex)

            # currentPoint = self.pointsOpen3D.points[currentPointIndex]
            pointNeighborhoodObject = NeighborsFactory.GetPointNeighborsByCoordinates(pointset3d, point=currentPoint,
                                                                                      searchRadius=searchRadius,
                                                                                      maxNN=maxNN,
                                                                                      useOriginal=useOriginal)
            neighborsProperty.setNeighbor(currentPointIndex, pointNeighborhoodObject)
            if rotate:
                neighborsProperty.RotatePointNeighborhood(currentPointIndex, smoothen=False, useOriginal=useOriginal)

        if returnValues:
            if len(idx) == 1:
                return pointNeighborhoodObject
            return pointNeighborhoodObject

    @staticmethod
    def GetPointNeighborsByCoordinates(pointset_open3d, point, searchRadius, maxNN, useOriginal=False):
        """
        Define a point's neighborhood according to its coordinates

        :param pointset_open3d: the point cloud in which the neighbors should be searched
        :param point: x, y, z of the point
        :param searchRadius: the radius in which the neighbors should be searched
        :param maxNN: the maximum number of neighbors
        :param useOriginal: ???

        :type pointset_open3d: PointSetOpen3D
        :type point: np.ndarray or tuple
        :type searchRadius: float
        :type maxNN: int
        :type useOriginal: bool

        :return: the neighborhood of a point
        
        :rtype: PointNeighborhood
        """

        if maxNN <= 0:
            if not useOriginal:
                num, idx, dist = pointset_open3d.kdTreeOpen3D.search_radius_vector_3d(point, radius=searchRadius)
            else:
                num, idx, dist = pointset_open3d.originalkdTreeOpen3D.search_radius_vector_3d(point,
                                                                                              radius=searchRadius)

        elif searchRadius <= 0:
            if not useOriginal:
                num, idx, dist = pointset_open3d.kdTreeOpen3D.search_knn_vector_3d(point, knn=maxNN)
            else:
                num, idx, dist = pointset_open3d.originalkdTreeOpen3D.search_knn_vector_3d(point, knn=maxNN)

        else:
            if not useOriginal:
                num, idx, dist = pointset_open3d.kdTreeOpen3D.search_hybrid_vector_3d(point, radius=searchRadius,
                                                                                      max_nn=maxNN)
            else:
                num, idx, dist = pointset_open3d.originalkdTreeOpen3D.search_hybrid_vector_3d(point,
                                                                                              radius=searchRadius,
                                                                                              max_nn=maxNN)
        pointsubset = PointSubSetOpen3D(pointset_open3d, idx)
        pointNeighborhood = PointNeighborhood(searchRadius, maxNN, pointsubset, dist)
        return pointNeighborhood

    @staticmethod
    def CalculateAllPointsNeighbors(pointset, searchRadius=0.05, maxNN=20, maxProcesses=8):
        # threadsList = []
        # pointsIndices = np.arange(self.numberOfPoints)
        # splitIndices = np.array_split(pointsIndices, maxProcesses)
        # p = Pool(maxProcesses)
        # filledFunction = partial(self.GetPointsNeighborsByID, searchRadius=searchRadius, maxNN=maxNN,
        #                          returnValues=False)
        # p.map(filledFunction, splitIndices)
        # p.map(self.GetPointsNeighborsByID, splitIndices, [searchRadius] * maxProcesses, [maxNN] * maxProcesses,
        # [False] * maxProcesses)
        # print("Done??????")

        # Function will be used with multiprocessing.
        # To run it without:
        if not isinstance(pointset, PointSetOpen3D):
            pointset = PointSetOpen3D(pointset)

        pointsIndices = np.arange(pointset.Size)
        neighbors = NeighborsProperty(pointset)
        NeighborsFactory.GetPointNeighborsByID(pointset, pointsIndices, searchRadius, maxNN, False)
        return neighbors



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

        pointsInRange = PointNeighborhood(radius, None, None, indices, dists)

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
        l = tree.query(pnt, pSize, p = 2, distance_upper_bound = radius)
        #         neighbor = PointSubSet(pntSet, l[1][where(l[0] != inf)[0]])
        neighbor = l[1][where(l[0] != inf)[0]]
        return PointSubSet(pntSet, neighbor), tree

    @staticmethod
    def GetNeighborsIn3dRange_BallTree(pnt, pntSet, radius, tree = None, num_neighbor = None):
        '''
        Find neighbors of a point using KDtree
        

        :param pnt: search point coordinates
        :param pntSet: pointset - in cartesian coordinates
        :param radius: search radius
        :param tree: ball tree
        :param num_neighbor: number of nearest neighbors
                if num_neighbor!=None the result will be the exact number of neighbors 
                and not neighbors in radius

        :type pnt: tuple?
        :type pntSet: PointSet.PointSet
        :type radius: float
        :type tree: BallTree
        :type num_neighbor: int

        :return: neighbors

        :rtype: PointSubSet

        '''

        if num_neighbor == None:
            pSize = pntSet.Size
        else:
            pSize = num_neighbor

        if tree == None:
            tree = BallTree(pntSet.ToNumpy())

        ind = tree.query_radius(pnt, r = radius)
        neighbor = PointSubSet(pntSet, ind[0])
        return neighbor

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
            hstack((expand_dims(points.Azimuths, 1), expand_dims(points.ElevationAngles, 1))))

        indices = nonzero(pntInCell)
        i1 = where(points.Ranges[indices[0]] >= pnt[2] - 0.10)
        i2 = where(points.Ranges[indices[0][i1[0]]] <= pnt[2] + 0.10)
        neighbors = SphericalCoordinatesFactory.CartesianToSphericalCoordinates(
            PointSubSet(points.XYZ, indices[0][i1[0][i2[0]]]))
        return neighbors

    @staticmethod
    def __PrintOverrideNeighborhoodCalculations(neighborProperty, exampleIndex, newRadius, newMaxNN):
        """

        :param neighborProperty: an existing NeighborsProperty which will be override
        :param exampleIndex:
        :param newRadius:
        :param newMaxNN:

        :type neighborProperty: NeighborsProperty
        :type exampleIndex:
        :type newRadius: float
        :type newMaxNN: int


                """
        previousRadius = neighborProperty.getNeighbors(exampleIndex).GetRadius()
        previousMaxNN = neighborProperty.getNeighbors(exampleIndex).GetMaxNN()

        if previousRadius != newRadius or previousMaxNN != newMaxNN:
            print("Function: PointSetOpen3D.PointSetOpen3D.GetPointsNeighborsByID")
            print("Overriding Previous Calculations")

            print("Previous Radius/maxNN: " + str(previousRadius) + "/" + str(previousMaxNN))
            print("New Radius/maxNN:\t" + str(newRadius) + "/" + str(newMaxNN))
            print()
