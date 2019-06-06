from numpy import array, nonzero, dot, logical_and, int_, cross, log, exp, isnan
from numpy.linalg import norm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from BallTreePointSet import BallTreePointSet


class TensorConnectivityGraph(object):

    def __init__(self, tensors, numNeighbors, varianceThreshold, normalSimilarityThreshold, distanceThreshold,
                 mode='binary', linearityThreshold=5):
        """
        Constructing a sparse similarity graph between the given tensors
        :param tensors: list of tensors (list/ndarray of Tensor objects)
        :param numNeighbors: the number of neighbors to find for each tensor (int)
        :param varianceThreshold: max allowed variance of a surface tensor (float)
        :param normalSimilarityThreshold: max difference allowed between the directions of the tensors,
        given in values of the angle sine (float)
        :param distanceThreshold: max distance between two tensors (float)
        :param mode: similarity function indicator. Valid values include: 'binary' (default), 'soft_clipping' and 'exp'
        :param linearityThreshold: the value between the two largest eigenvalues from which a tensor is considered as
        a linear one (float)
        """
        # parsing data from tensors
        self.__cogs = array(list(map(lambda t: t.reference_point, tensors)))
        self.__eigVals = array(list(map(lambda t: t.eigenvalues, tensors)))
        self.__stickAxes = array(list(map(lambda t: t.stick_axis, tensors)))
        self.__plateAxes = array(list(map(lambda t: t.plate_axis, tensors)))

        # finding neighbors for each tensor
        cogsBallTree = BallTreePointSet(self.__cogs, leaf_size=20)
        self.__neighbors = cogsBallTree.query(self.__cogs, numNeighbors + 1)[:, 1:]

        self.__varianceThreshold = varianceThreshold
        self.__normalSimilarityThreshold = normalSimilarityThreshold
        self.__distanceThreshold = distanceThreshold
        self.__linearityThreshold = linearityThreshold
        self.__labels = None

        # initializing similarity matrix
        numTensors = len(tensors)
        self.__simMatrix = coo_matrix((numTensors, numTensors), dtype='f').tolil()
        self.__simMatrix[range(numTensors), range(numTensors)] = 1

        # filtering noisy tensors
        validNodes = nonzero(self.__eigVals[:, 0] < self.__varianceThreshold)[0]

        # computing similarity between neighboring tensors
        list(map(lambda i: self.__computeSimilarityForTensor(i, mode), validNodes))

    def __computeSimilarityForTensor(self, index, mode='binary'):
        """
        Computing the similarity value between a tensor given by its index and its neighbors
        :param index: index of the tensor (int)
        :param mode: similarity function indicator. Valid values include: 'binary' (default), 'soft_clipping' and 'exp'
        :return: None
        """
        lambda2 = self.__eigVals[index, 1]
        eigRatio = self.__eigVals[index, -1] / self.__eigVals[index, 1]
        neighbors = self.__neighbors[index]
        neighbors = neighbors[self.__eigVals[neighbors, 0] < self.__varianceThreshold]

        deltas = self.__cogs[neighbors] - self.__cogs[index]

        if eigRatio < self.__linearityThreshold or lambda2 > self.__varianceThreshold:
            distances = abs(dot(self.__plateAxes[index].reshape((1, 3)), deltas.T)).reshape((-1,))
            directionalDiffs = abs(dot(self.__plateAxes[index].reshape((1, 3)),
                                       self.__plateAxes[neighbors].T)).reshape((-1,))
        else:
            distances = norm(cross(self.__stickAxes[index], deltas), axis=1)
            directionalDiffs = abs(dot(self.__stickAxes[index].reshape((1, 3)),
                                       self.__stickAxes[neighbors].T)).reshape((-1,))

        self.__simMatrix[index, neighbors] = self.__computeEdgeWeight(distances, directionalDiffs, mode)

    def __computeEdgeWeight(self, distances, directionalDiffs, mode='binary'):
        """
        Computing the similarity values based on the distances and angular difference between a tensor and its neighbor
        based on a given similarity function
        :param distances: distance between tensor and its neighbors (1-D ndarray)
        :param directionalDiffs: angular differences between tensor and its neighbors (1-D ndarray)
        :param mode: similarity function indicator. Valid values include: 'binary' (default), 'soft_clipping' and 'exp'
        :return: similarity values (1-D ndarray)
        """
        if mode == 'binary':
            distanceTest = distances < self.__distanceThreshold
            directionTest = directionalDiffs > 1 - self.__normalSimilarityThreshold
            return int_(logical_and(distanceTest, directionTest))
        elif mode == 'soft_clipping':
            distAlpha = 100  # TODO: ADD AS PARAMETER
            ds = distances / self.__varianceThreshold
            ds[ds > 5] = 5
            distanceWeights = 1 - 1 / distAlpha * log((1 + exp(distAlpha * ds)) / (1 + exp(distAlpha * (ds - 1))))
            distanceWeights[isnan(distanceWeights)] = 0

            angAlpha = 10  # TODO: ADD AS PARAMETER
            angs = (directionalDiffs - (1 - self.__normalSimilarityThreshold)) / self.__normalSimilarityThreshold
            angWeights = 1 / angAlpha * log((1 + exp(angAlpha * angs)) /
                                            (1 + exp(angAlpha * (angs - 1))))

            return distanceWeights * angWeights
        elif mode == 'exp':
            return exp(-distances ** 2 / self.__distanceThreshold) * \
                   exp(directionalDiffs ** 2 / self.__normalSimilarityThreshold)
        else:
            raise ValueError('Unrecognised weighting method')

    def connected_componnents(self):
        """
        Graph partitioning by the Connected Components method
        :return: tuple containing the number of segments the graph was partitioned to, labels of each node in graph,
        and a list containing the list of nodes per segment
        """
        numComponnets, labels = connected_components(self.__simMatrix)
        self.__labels = labels
        indexesByLabels = [nonzero(labels == l)[0] for l in range(numComponnets)]
        return numComponnets, labels, indexesByLabels

    def spyGraph(self):
        """
        Plots the connectivity graph. used for debugging purposes.
        :return: None
        """
        from matplotlib import pyplot as plt
        plt.spy(self.__simMatrix)

    def getSimilarityValue(self, index1, index2):
        """
        Retrieves the similarity value between two nodes in the connectivity graph
        :param index1: index of 1st node (int)
        :param index2: index of 2nd node (int)
        :return: similarity value (float)
        """
        return self.__simMatrix[index1, index2]


if __name__ == '__main__':
    pass
