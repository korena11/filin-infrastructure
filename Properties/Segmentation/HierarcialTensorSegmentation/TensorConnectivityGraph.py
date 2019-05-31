from numpy import array, nonzero
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class TensorConnectivityGraph(object):

    def __init__(self, tensors, neighbors, varianceThreshold, normalSimilarityThreshold, distanceThreshold=None,
                 mode='binrary', linearityThreshold=5):
        # self.__tensors = tensors
        self.__cogs = array(list(map(lambda t: t.reference_point, tensors)))
        self.__eigVals = array(list(map(lambda t: t.eigenvalues, tensors)))
        self.__stickAxes = array(list(map(lambda t: t.stick_axis, tensors)))
        self.__plateAxes = array(list(map(lambda t: t.plate_axis, tensors)))

        self.__neighbors = neighbors
        self.__varianceThreshold = varianceThreshold
        self.__normalSimilarityThreshold = normalSimilarityThreshold
        self.__distanceThreshold = distanceThreshold
        self.__linearityThreshold = linearityThreshold
        self.__labels = None

        numTensors = len(tensors)
        self.__simMatrix = coo_matrix((numTensors, numTensors), dtype='f').tolil()
        list(map(lambda i: self.__computeSimilarityForTensor(i, mode), range(numTensors)))

    def __computeSimilarityForTensor(self, index, mode='binary'):
        self.__simMatrix[index, index] = 1  # TODO: Move to __init__
        lambda3 = self.__eigVals[index, 0]
        lambda2 = self.__eigVals[index, 1]
        eigRatio = self.__eigVals[index, -1] / self.__eigVals[index, 1]
        neighbors = self.__neighbors[index]

        if lambda3 < self.__varianceThreshold:  # TODO: Move to __init__, filter nodes that fail test
            deltas = cogs[neighbors] - cogs[index]

    def connected_componnents(self):
        """
        Graph partitioning by the Connected Components method
        :return: tuple containing the number of segments the graph was partitioned to and a list of the segments
        """
        numComponnets, labels = connected_components(self.__simMatrix)
        self.__labels = labels
        return numComponnets, labels

    def __createSegment(self, keys, tensors):
        """

        :param keys:
        :param tensors:
        :return:
        """
        # TODO: Implement TensorSet Class
        return None
        # s = Segment()
        # list(map(s.addPatch, keys, tensors))
        # return s

    def connected_segments(self):
        """

        :return:
        """
        numComponnets, labels = connected_components(self.__simMatrix)
        labeledIndexes = [nonzero(labels == l)[0] for l in range(numComponnets)]
        # labeledKeys = list(map(array(list(self.__tensors.keys())).__getitem__, labeledIndexes))
        labeledTensors = list(map(array(self.__tensors).__getitem__, labeledIndexes))
        return None
        # return list(map(self.__createSegment, labeledIndexes, labeledTensors))

    def spyGraph(self):
        """
        Plots the connectivity graph
        :return:
        """
        from matplotlib import pyplot as plt
        plt.spy(self.__simMatrix)

    def getSimilarityValue(self, index1, index2):
        """
        Retrieves the similarity value between two nodes in the connectivity graph
        :param index1: index of 1st node (int)
        :param index2: index of 2nd node (int)
        :return:
        """
        return self.__simMatrix[index1, index2]


if __name__ == '__main__':
    pass
