from numpy import array, nonzero, dot, logical_and, int_, cross, log, exp, isnan, hstack, unique, zeros, cumsum
from numpy.linalg import norm
from scipy.stats import chi2
from scipy.sparse import coo_matrix, find
from scipy.sparse.csgraph import connected_components
from matplotlib import pyplot as plt

from BallTreePointSet import BallTreePointSet


class TensorConnectivityGraph(object):

    def __init__(self, tensors, numNeighbors, varianceThreshold, normalSimilarityThreshold, distanceThreshold,
                 linearityThreshold=5, mode='binary', distanceSoftClippingAlpha=100, angularSoftClippingAlpha=10,
                 softClippingThreshold=0):
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
        if self.__cogs.shape[0] > numNeighbors:
            self.__neighbors = cogsBallTree.query(self.__cogs, numNeighbors + 1)[:, 1:]
        else:
            # creating a fully connected graph in case of small dataset
            self.__neighbors = cogsBallTree.query(self.__cogs, self.__cogs.shape[0])[:, 1:]

        self.__varianceThreshold = varianceThreshold
        self.__normalSimilarityThreshold = normalSimilarityThreshold
        self.__distanceThreshold = distanceThreshold
        self.__linearityThreshold = linearityThreshold
        self.__labels = None

        # initializing similarity matrix
        numTensors = len(tensors)
        self.__simMatrix = coo_matrix((numTensors, numTensors), dtype='f').tolil()
        self.__simMatrix[range(numTensors), range(numTensors)] = 1

        # TODO: delete following decision on paper methodology
        self.__disMatrix = coo_matrix((numTensors, numTensors), dtype='f').tolil()
        self.__angMatrix = coo_matrix((numTensors, numTensors), dtype='f').tolil()
        self.__donMatrix = coo_matrix((numTensors, numTensors), dtype='f').tolil()

        # filtering noisy tensors
        validNodes = nonzero(self.__eigVals[:, 0] < self.__varianceThreshold)[0]

        self.__mode = mode
        self.__distanceSoftClippingAlpha = distanceSoftClippingAlpha
        self.__angularSoftClippingAlpha = angularSoftClippingAlpha
        self.__softClippingThreshhold = softClippingThreshold

        # computing similarity between neighboring tensors
        list(map(self.__computeSimilarityForTensor, validNodes))

    def __computeSimilarityForTensor(self, index):
        """
        Computing the similarity value between a tensor given by its index and its neighbors
        :param index: index of the tensor (int)
        :param mode: similarity function indicator. Valid values include: 'binary' (default), 'soft_clipping' and 'exp'
        :return: None
        """
        lambda2 = self.__eigVals[index, 1] ** 0.5
        eigRatio = self.__eigVals[index, -1] / self.__eigVals[index, 1]
        neighbors = self.__neighbors[index]
        neighbors = neighbors[self.__eigVals[neighbors, 0] < self.__varianceThreshold]

        deltas = self.__cogs[neighbors] - self.__cogs[index]

        if eigRatio < self.__linearityThreshold or lambda2 > self.__varianceThreshold:
            distances = abs(dot(self.__plateAxes[index].reshape((1, 3)), deltas.T)).reshape((-1,))
            directionalDiffs = dot(self.__plateAxes[index].reshape((1, 3)),
                                   self.__plateAxes[neighbors].T).reshape((-1,))
            don = self.__plateAxes[neighbors] - self.__plateAxes[index]
            don[directionalDiffs < 0, :] = self.__plateAxes[neighbors][directionalDiffs < 0] + self.__plateAxes[index]
        else:
            distances = norm(cross(self.__stickAxes[index], deltas), axis=1)
            directionalDiffs = dot(self.__stickAxes[index].reshape((1, 3)),
                                   self.__stickAxes[neighbors].T).reshape((-1,))

            # TODO: VERIFY CORRECTNESS AND ThRESHOLD FOR LINEAR OBJECTS
            don = self.__stickAxes[neighbors] - self.__stickAxes[index]  # zeros((neighbors.shape[0], 3)) + 1e-16
            don[directionalDiffs < 0, :] = self.__stickAxes[neighbors][directionalDiffs < 0] + self.__stickAxes[index]

        self.__simMatrix[index, neighbors] = self.__computeEdgeWeight(distances, norm(don, axis=1) ** 2)
        self.__disMatrix[index, neighbors] = distances + 1e-16
        self.__angMatrix[index, neighbors] = abs(directionalDiffs)
        self.__donMatrix[index, neighbors] = norm(don, axis=1) ** 2

    def __computeEdgeWeight(self, distances, directionalDiffs):
        """
        Computing the similarity values based on the distances and angular difference between a tensor and its neighbor
        based on a given similarity function
        :param distances: distance between tensor and its neighbors (1-D ndarray)
        :param directionalDiffs: angular differences between tensor and its neighbors (1-D ndarray)
        :param mode: similarity function indicator. Valid values include: 'binary' (default), 'soft_clipping' and 'exp'
        :return: similarity values (1-D ndarray)
        """
        if self.__mode == 'binary':
            distanceTest = distances < self.__distanceThreshold
            # directionTest = directionalDiffs > 1 - self.__normalSimilarityThreshold
            directionTest = directionalDiffs < self.__normalSimilarityThreshold
            return int_(logical_and(distanceTest, directionTest))
        elif self.__mode == 'soft_clipping':
            distAlpha = self.__distanceSoftClippingAlpha
            ds = distances / self.__varianceThreshold
            ds[ds > 5] = 5
            distanceWeights = 1 - 1 / distAlpha * log((1 + exp(distAlpha * ds)) / (1 + exp(distAlpha * (ds - 1))))
            distanceWeights[isnan(distanceWeights)] = 0

            angAlpha = self.__angularSoftClippingAlpha
            angs = (directionalDiffs - (1 - self.__normalSimilarityThreshold)) / self.__normalSimilarityThreshold
            angWeights = 1 / angAlpha * log((1 + exp(angAlpha * angs)) /
                                            (1 + exp(angAlpha * (angs - 1))))

            edgeWeights = distanceWeights * angWeights
            edgeWeights[edgeWeights < self.__softClippingThreshhold] = 0
            return edgeWeights
        elif self.__mode == 'exp':
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
        self.__indexesByLabels = [nonzero(labels == l)[0] for l in range(numComponnets)]
        return numComponnets, labels, self.__indexesByLabels

    def collapseConnectivity(self):
        """

        :return:
        """
        tmp = list(map(lambda l: hstack(list(map(self.__neighbors.__getitem__, l))), self.__indexesByLabels))
        labelledNeighbors = list(map(unique, list(map(self.__labels.__getitem__, tmp))))
        labelledNeighbors = array(list(map(lambda l: labelledNeighbors[l][labelledNeighbors[l] != l],
                                           range(len(labelledNeighbors)))))

        return labelledNeighbors


    def spyGraph(self):
        """
        Plots the connectivity graph. used for debugging purposes.
        :return: None
        """
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
