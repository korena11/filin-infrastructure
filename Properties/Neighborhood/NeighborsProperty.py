import numpy as np
from tqdm import tqdm

from Properties.BaseProperty import BaseProperty
from Properties.Neighborhood.PointNeighborhood import PointNeighborhood




class NeighborsProperty(BaseProperty):
    """
    Property that holds the neighbors of each point
    """

    def __init__(self, points, **kwargs):
        """

        :param points:

        :type points: PointSet, PointSetOpen3D
        :param args:


        .. warning::
            Now works only for Linux (or with open3D for windows)
        """
        super(NeighborsProperty, self).__init__(points)
        self.__pointsNeighborsArray = np.empty(shape=(self.Size,), dtype=PointNeighborhood)
        self.__averageRadius = None
        self.__averageNeighborsNumber = None

    def __next__(self):
        self.current += 1
        try:
            return self.getNeighborhood(self.current - 1)
        except IndexError:
            self.current = 0
            raise StopIteration

    def getValues(self):
        """
        All neighbors of all computed points

        :return:
        """
        return self.__pointsNeighborsArray

    def getNeighbors(self, idx):
        """
        Retrieve the neighbors of point(s) with index

        :param idx: the index of the point

        :type idx: int, tuple, list

        :return: the tensor of point idx

        :rtype: PointSubSet or PointSubSetOpen3D
        """

        return self.getNeighborhood(idx).neighbors

    def getNeighborhood(self, idx):
        """
        Retrieve the point neighborhood

        :rtype: PointNeighborhood
        """
        neighbors = self.__pointsNeighborsArray[idx]

        if neighbors is None:
            return neighbors

        # elif np.all(neighbors.neighborhoodIndices == idx):
        #     neighbors = None

        return neighbors

    def setNeighborhood(self, idx, point_neighbors):
        """
        Set a PointNeighborhood into the property according to the point index

        :param idx: the index (or indices) of the point(s) to set
        :param point_neighbors: the PointeNeighborhood object to set

        :type idx: int, np.array
        :type point_neighbors: PointNeighborhood

        """
        if isinstance(point_neighbors, PointNeighborhood):
            self.__pointsNeighborsArray[idx] = point_neighbors

        else:
            if isinstance(point_neighbors, np.ndarray):
                if isinstance(point_neighbors[0], PointNeighborhood):
                    self.__pointsNeighborsArray = point_neighbors
            else:
                subset = point_neighbors.neighbors
                self.__pointsNeighborsArray[idx] = PointNeighborhood(subset)




    def average_neighborhood_radius(self):
        """
        compute the average radius of all the point neighborhood in the class

        :return: the average radius

        :rtype: float
        """
        if self.__averageRadius is None:
            radii = []
            for point_neighborhood in self.__pointsNeighborsArray:
                radii.append(point_neighborhood.radius)

            self.__averageRadius = np.asarray(radii).mean()
        return self.__averageRadius

    def average_neighborhood_size(self):
        """
        compute the average number of neighbors of all the point neighborhood in the class

        :return: the average radius

        :rtype: float
        """
        if self.__averageNeighborsNumber is None:
            size = []
            for point_neighborhood in self.__pointsNeighborsArray:
                size.append(point_neighborhood.numberOfNeighbors)

            self.__averageNeighborsNumber = int(np.asarray(size).mean())
        return self.__averageNeighborsNumber

    @property
    def ToCUDA(self):
        """
        changing NeighborProperty format for using with CUDA functionality
        :return:
        """

        if isinstance(self.__pointsNeighborsArray[0], PointNeighborhood):
            neighborsCount = np.array(list(map(lambda n: n.Size - 1,

            # getting number of neighboring points
            neighborsCount = np.array(list(map(lambda n: n.Size - 1,
                                               tqdm(self.__pointsNeighborsArray,
                                                    'Getting number of neighbors for each point'))))
            neighbors = list(map(lambda n: n.neighbors.ToNumpy()[1:].reshape((-1,)),

            # getting list of neighbors per point (without the point itself)
            neighbors = list(map(lambda n: n.neighbors.ToNumpy()[1:].reshape((-1, )),
                                 tqdm(self.__pointsNeighborsArray, desc='Retrieving neighbors')))

            maxNeighbors = neighborsCount.max()  # getting the number of maximum point

            neighbors = np.array(list(map(
                lambda n: n if n.shape[0] == 3 * maxNeighbors
                else np.hstack([n, np.zeros((3 * maxNeighbors - n.shape[0], ))]),
                tqdm(neighbors, desc='Zero-padding neighbors array'))))

        elif isinstance(self.__pointsNeighborsArray[0], np.ndarray):
            # TODO: convert numpy array
            return None
        else:
            raise TypeError('Cannot convert property for CUDA')

        return neighbors, neighborsCount
