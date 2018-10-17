import numpy as np
from scipy.spatial import Delaunay


class PointNeighborhood:
    def __init__(self, radius, max_neighbor_num, num_pts, idx, dist):
        """
        :param radius: Radius of neighborhood
        :param max_neighbor_num: Max number of neighborhood points set
        :param num_pts: Number of points in neighborhood
        :param idx: Neighborhood points indices
        :param dist: Distance of neighborhood points from center point
        """
        # self.pointSet = super(PointNeighborhood, self).__init__(points, idx)

        self.r = radius
        self.nn = max_neighbor_num
        self.num = num_pts
        self.idx = idx
        self.dist = dist

        self.localRotatedNeighbors = None

    @property
    def radius(self):
        return self.r

    @property
    def maxNN(self):
        return self.nn

    @property
    def numberOfNeighbors(self):
        return self.num

    @property
    def neighborhoodIndices(self):
        return self.idx

    @property
    def localRotatedNeighborhood(self):
        return self.localRotatedNeighbors

    @localRotatedNeighborhood.setter
    def localRotatedNeighborhood(self, neighborsArray):
        self.localRotatedNeighbors = neighborsArray.copy()

    def VisualizeNeighborhoodTriangulation(self):
        """
        .. warning::

            Not working

        :return:
        """
        flag = 0
        idx = np.arange(len(self.localRotatedNeighbors))

        # simplices returns points IDs for each triangle
        triangulation = (Delaunay(self.localRotatedNeighbors[:, 0:2])).simplices

        tri = np.where(triangulation == 0)[0]  # Keep triangles that have the first point in them
