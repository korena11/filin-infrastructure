import numpy as np
from scipy.spatial import Delaunay

from PointSubSet import PointSubSet
from PointSubSetOpen3D import PointSubSetOpen3D


class PointNeighborhood:
    def __init__(self, radius, max_neighbor_num, points_subset, dist=None):
        """
        :param radius: Radius of neighborhood
        :param max_neighbor_num: Max number of neighborhood points set
        :param points_subset: the neighborhood as point subset
        :param dist: Distance of neighborhood points from center point


        :type radius: float
        :type max_neighbor_num: int
        :type points_subset: PointSubSet or PointSubSetOpen3D
        """

        self.r = radius
        self.nn = max_neighbor_num
        self.dist = dist

        self.__neighbors = points_subset

    @property
    def radius(self):
        return self.r

    @property
    def maxNN(self):
        return self.nn

    @property
    def numberOfNeighbors(self):
        return self.__neighbors.Size

    @property
    def neighborhoodIndices(self):
        return self.__neighbors.GetIndices

    @property
    def neighbors(self):
        return self.neighbors

    @neighbors.setter
    def neighbors(self, pointsubset):
        self.__neighbors = pointsubset

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
