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

            don = zeros((neighbors.shape[0], 3)) + 1e-16
            # don[directionalDiffs < 0, :] = self.__stickAxes[neighbors][directionalDiffs < 0] + self.__stickAxes[index]

        self.__simMatrix[index, neighbors] = 1   # self.__computeEdgeWeight(distances, abs(directionalDiffs))
        self.__disMatrix[index, neighbors] = distances
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
            directionTest = directionalDiffs > 1 - self.__normalSimilarityThreshold
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

    def nullifyEdges(self, p=0.99):
        """

        :param p:
        :return:
        """
        r, c, dis = find(self.__disMatrix)
        disStd = dis.sum() / dis.shape[0]
        # disCounts, disBins, _ = plt.hist(dis, bins='auto')
        # disBins = array(list(map(lambda i: 0.5 * (disBins[i] + disBins[i + 1]), range(disBins.shape[0] - 1))))
        # disFreqs = disCounts / disCounts.sum()
        # disThr = disBins[nonzero(cumsum(disFreqs) <= p)[0][-1]]
        disThr = chi2.ppf(q=p, df=1, scale=disStd)
        self.__simMatrix[r[dis > disThr], c[dis > disThr]] = 0

        r, c, don = find(self.__donMatrix)
        donStd = don.sum() / don.shape[0]
        # donCounts, donBins, _ = plt.hist(don, bins='auto')
        # donBins = array(list(map(lambda i: 0.5 * (donBins[i] + donBins[i + 1]), range(donBins.shape[0] - 1))))
        # donFreqs = donCounts / donCounts.sum()
        # donThr = donBins[nonzero(cumsum(donFreqs) <= p)[0][-1]]
        donThr = chi2.ppf(q=p, df=2, scale=donStd)
        self.__simMatrix[r[don > donThr], c[don > donThr]] = 0

        # temp = nonzero(logical_and(self.__disMatrix < disThr, self.__donMatrix < donThr))[0]
        # self.__simMatrix[temp[0], temp[1]] = 0

        # plt.figure()
        # plt.title('Distances histogram')
        # plt.hist(dis, bins=100)
        # plt.yscale('log')
        #
        # # plt.figure()
        # # plt.title('direction dot product histogram')
        # # plt.hist(ang, bins=100)
        # # plt.yscale('log')
        #
        # plt.figure()
        # plt.title('difference of normals histogram')
        # plt.hist(don, bins=100)
        # plt.yscale('log')
        #
        # # plt.figure()
        # # plt.scatter(don, ang)
        # # plt.xlabel('squared norm of difference of normals')
        # # plt.ylabel('cosine of angle between normals')
        #
        # plt.show()

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
