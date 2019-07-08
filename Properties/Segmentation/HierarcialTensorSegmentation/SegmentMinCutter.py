from numpy import array, hstack, vstack, int_, exp
from numpy.linalg import norm
from BallTreePointSet import BallTreePointSet
from networkx import DiGraph, minimum_cut

class SegmentMinCutter(object):

    def __init__(self, segment1, segment2, numNeighbors=4, offset=None):
        self.__segment1 = segment1
        self.__segment2 = segment2

        cogs1 = array(list(map(lambda t: t.reference_point, segment1.tensors)))
        cogs2 = array(list(map(lambda t: t.reference_point, segment2.tensors)))

        ballTree1 = BallTreePointSet(cogs1)
        ballTree2 = BallTreePointSet(cogs2)

        neighbors11 = ballTree1.query(cogs1, numNeighbors + 1)[:, 1:]
        neighbors12 = ballTree2.query(cogs1, numNeighbors) + cogs1.shape[0]
        neighobors1 = list(map(hstack, list(zip(neighbors11, neighbors12))))

        neighbors21 = ballTree1.query(cogs2, numNeighbors)
        neighbors22 = ballTree2.query(cogs2, numNeighbors + 1)[:, 1:] + cogs1.shape[0]
        neighobors2 = list(map(hstack, list(zip(neighbors21, neighbors22))))

        self.__cogs = vstack([cogs1, cogs2])
        self.__neighbors = vstack([neighobors1, neighobors2])
        self.__tensors = hstack([segment1.tensors, segment2.tensors])
        if offset is None:
            self.__offset = max(segment1.eigenvalues[0], segment2.eigenvalues[0]) ** 0.5
        else:
            self.__offset = offset

        self.__numNodes = self.__cogs.shape[0]
        self.__graph = DiGraph()
        list(map(self.__addEdges, range(self.__numNodes)))

    def __addEdges(self, index):
        """
        Adding weighted edges to the graph from a given node based on euclidean distances its neighbors
        :param index: index of node to add edges for
        :return: None
        """
        neighobrs = self.__neighbors[index]
        dists = norm(self.__cogs[neighobrs] - self.__cogs[index], axis=1) ** -2  # TODO: compute without norm

        list(map(lambda neighbor, dist: self.__graph.add_edge(index, neighbor, capacity=dist), neighobrs, dists))

    def minCut(self):
        """

        :return:
        """
        distFromSegment1 = self.__segment1.distanceFromPoint(self.__cogs) ** -2
        distFromSegment2 = (self.__segment2.distanceFromPoint(self.__cogs) + self.__offset) ** -2

        list(map(lambda i: self.__graph.add_edge('source', i, capacity=distFromSegment1[i]), range(self.__numNodes)))
        list(map(lambda i: self.__graph.add_edge(i, 'sink', capacity=distFromSegment2[i]), range(self.__numNodes)))

        cutValue, partition = minimum_cut(self.__graph, 'source', 'sink')
        reachable, notReachable = partition

        reachable = array(list(reachable))
        intReachable = int_(reachable[reachable != 'source'])

        notReachable = array(list(notReachable))
        intNotReachable = int_(notReachable[notReachable != 'sink'])

        if (len(intReachable) == 0 and len(intNotReachable) == self.__numNodes) or \
                (len(intReachable) == self.__numNodes and len(intNotReachable) == 0):
            return True
        else:
            return False


if __name__ == '__main__':
    pass
