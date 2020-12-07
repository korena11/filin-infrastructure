import numpy as np

from DataClasses.PointSubSet import PointSubSet


class PointNeighborhood(object):


    def __init__(self, points_subset, distances=None):
        """
        Removes duplicate points, if exist (keeps the first)

        :param points_subset: the neighborhood as point subset
        :param distances: Distances of each point from center point (computed unless sent)

        :type points_subset: PointSubSet, PointSubSetOpen3D
        :type distances: np.array

        """
        self.__neighbors = points_subset
        if distances is None:
            self.computeDistances()
        else:
            self.__distances = distances
        self.__weights = np.ones((self.Size, ))

    @property
    def weights(self):
        """
        Neighborhood weighting.

        :return: the weight of each point (according to distances)

        :rtype: float, np.array
        """
        return self.__weights

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
    def weighted_distances(self):
        """
        Weighted distances according to the neighborhood weights

        :return: distances multiplied by their weight

        :rtype: np.array, float
        """
        return self.distances * self.weights

    @property
    def numberOfNeighbors(self):
        """
        The number of  neighbors (without the point itself)

        :return: the number of neighbors

        :rtype: int
        """
        return self.__neighbors.Size - 1

    @property
    def neighborhoodIndices(self):
        return self.__neighbors.GetIndices

    @property
    def Size(self):
        """
        Return the size of the subset (with the center point)
        """
        return self.numberOfNeighbors

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
        from DataClasses.PointSubSetOpen3D import PointSubSetOpen3D

        center_pt = self.center_point_coords
        pts = self.__neighbors.ToNumpy()

        distances = np.linalg.norm(pts - center_pt, axis=1)

        if np.nonzero(distances == 0)[0].shape[0] > 1:

            # print('Point set has two identical points at {}'.format(center_pt))
            tmp_subset = PointSubSet(self.neighbors.data, np.hstack((self.center_point_idx, self.neighborhoodIndices[np.nonzero(distances != 0)])))
            self.__init__(tmp_subset)
            pts = self.__neighbors.ToNumpy()
            distances = np.linalg.norm(pts - center_pt, axis=1)

        self.__distances = distances
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

        return directions[np.nonzero(self.__distances != 0)] / self.__distances[np.nonzero(self.__distances != 0)][:,
                                                               None]

    def weightNeighborhood(self, weightingFunc, *args):
        """
        Compute weights to a neighborhood according to a weightingFunc that is sent.

        :param weightingFunc: weighting function (can be taken from WeightingFunctions module.
        :param kwargs: according to the sent function:
            - WeightingFunctions.triangleWeights(self, effectiveDistance)


        .. seealso::
           `Properties.Neighborhood.WeightingFunctions.triangleWeights`

        """
        import Properties.Neighborhood.WeightingFunctions as wf

        weights = weightingFunc(self, *args)
        self.__weights = weights

    # --------------- I THINK THIS IS REDUNDANT. CONSIDER REMOVING ----------------------------
    def color_neighborhood(self, point_color='red', neighbors_color='black'):
        """
        Assign point_color color to the center point and neighbor_color to the rest

        :param point_color: name or rgb of the center point. Default: 'red'
        :param neighbors_color: name or rgb of neighbor points. Default: 'black'

        :type point_color: (str, tuple)
        :type neighbors_color: (str, tuple)
        :return: array with colors

        :rtype: ColorProperty.ColorProperty

        .. warning::
            REDUNDANT; CONSIDER REMOVING
        """
        import webcolors
        from Properties.Color.ColorProperty import ColorProperty
        if type(neighbors_color) is str:
            neighbors_color = webcolors.name_to_rgb(neighbors_color)

        if type(point_color) is str:
            point_color = webcolors.name_to_rgb(point_color)

        colors = np.ones((self.Size, 3)) * neighbors_color
        colors[0, :] = point_color

        return ColorProperty(self.__neighbors, colors)