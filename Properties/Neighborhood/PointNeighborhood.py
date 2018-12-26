import numpy as np

from PointSubSet import PointSubSet


class PointNeighborhood:
    def __init__(self, points_subset, distances):
        """

        :param points_subset: the neighborhood as point subset
        :param distances: Distances of each point from center point



        :type points_subset: PointSubSet, PointSubSetOpen3D
        :type distances: np.array

        """

        self.__distances = distances
        self.__neighbors = points_subset


    @property
    def radius(self):
        """
        Mean radius of the neighbors

        :return: mean radius
        :rtype: float
        """

        return np.mean(self.__distances)

    @property
    def numberOfNeighbors(self):
        """
        The number of points within the neighborhood (including the point itself)

        :return: the number of points within the neighborhood

        :rtype: int
        """
        return self.__neighbors.Size

    @property
    def neighborhoodIndices(self):
        return self.__neighbors.GetIndices

    @property
    def neighbors(self):
        """
        Return a point set of the neighborhood

        :return: points that compose the neighborhood (including the point itself at index 0)

        :rtype: PointSubSet
        """
        return self.__neighbors

    @neighbors.setter
    def neighbors(self, pointsubset):
        self.__neighbors = pointsubset

    @property
    def center_point_coords(self):
        """
        The point to which the neighbors relate

        :return: coordinates of the center point

        :rtype: np.ndarray
        """
        return self.neighbors.GetPoint(0)

    @property
    def center_point_idx(self):
        """
        The index of the point to which the neighbors relate

        :return: index of the center point

        :rtype: int
        """
        return self.neighbors.GetIndices[0]
