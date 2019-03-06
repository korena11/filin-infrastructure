import numpy as np

from PointSubSet import PointSubSet


class PointNeighborhood:
    def __init__(self, points_subset, distances=None):
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
        if self.__distances is None:
            self.computeDistances()

        return np.mean(self.__distances)

    @property
    def distances(self):
        """
        Array of the distances between each point and the center point

        :return: array of distances

        :rtype: np.array
        """
        if self.__distances is None:
            self.computeDistances()

        return self.__distances

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

    def computeDistances(self):
        """
        Compute the distances between each point and the center point

        :return: array of distances

        :rtype: np.array
        """

        center_pt = self.center_point_coords
        pts = self.__neighbors.ToNumpy()

        self.__distances = np.linalg.norm(pts - center_pt, axis=1)

        return self.__distances

    def neighbors_vectors(self):
        """
        Find the direction of each point to the center point

        :return: array of directions

        :rtype: np.array nx3
        """
        center_pt = self.center_point_coords
        pts = self.__neighbors.ToNumpy()

        directions = pts - center_pt

        if self.__distances is None:
            self.__distances = np.linalg.norm(directions, axis=1)
        if np.nonzero(self.__distances == 0)[0].shape[0] > 1:
            print('Point set has two identical points at {}'.format(center_pt))
        return directions[np.nonzero(self.__distances != 0)] / self.__distances[np.nonzero(self.__distances != 0)][:,
                                                               None]
