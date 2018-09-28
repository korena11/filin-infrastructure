from matplotlib.path import Path
from numpy import mean, round, nonzero, where, hstack, inf, rad2deg, expand_dims
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

from PointSet import PointSet
from PointSubSet import PointSubSet
from SphericalCoordinatesFactory import SphericalCoordinatesFactory


class NeighborsFactory:
    """
    Find neighbors of a given points using different methods  
    """

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

        indices = indices[indices > 0]

        pointsInRange = PointSubSet(points, indices)

        return pointsInRange

    @staticmethod
    def GetNeighborsIn3dRange_KDtree(pnt, pntSet, radius, tree = None, num_neighbor = None):
        '''
        Find neighbors of a point using KDtree

        :param pnt: search point coordinates
        :param pntSet: the points in which the neighbors are required - in cartesian coordinates
        :param radius: search radius
        :param tree: KD tree
        :param num_neighbor: number of nearest neighbors
                if num_neighbor!=None the result will be the exact number of neighbors 
                and not neighbors in radius
         
        :type pnt: tuple?
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
