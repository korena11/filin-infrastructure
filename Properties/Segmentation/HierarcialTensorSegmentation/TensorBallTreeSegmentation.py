import warnings

from numpy import zeros, array

from BallTreePointSet import BallTreePointSet
from PointSet import PointSet
from TensorConnectivityGraph import TensorConnectivityGraph
from TensorFactory import TensorFactory


def ExtractSurfaceElements(points, leafSize=10, smallestObjectSize=0.1):
    """

    :param points:
    :param leafSize:
    :param smallestObjectSize:
    :return:
    """

    if not isinstance(points, PointSet):
        raise TypeError('points is not a PointSet object')
    elif isinstance(points, BallTreePointSet):
        warnings.warn('Ball-tree is already constructed, skipping to the next step')
        ballTree = points
    else:
        # Constructing
        ballTree = BallTreePointSet(points, leaf_size=leafSize)

    # Getting the indices of the deepest nodes of minimal object size
    nodesOfInterest = ballTree.getSmallestNodesOfSize(smallestObjectSize)

    # Getting the lists of points for each initial segment
    pointsOfNodes = list(map(ballTree.getPointsOfNode, nodesOfInterest))

    tensors = list(map(TensorFactory.tensorFromPoints,
                       list(map(ballTree.ToNumpy().__getitem__, pointsOfNodes))))

    # Assigning initial labels to the points
    labels = zeros((ballTree.Size,), dtype='i')
    for i in range(len(tensors)):
        labels[pointsOfNodes[i]] = i
    return ballTree, labels, nodesOfInterest, tensors


def tensorConnectedComponents(tensors, numNeigbhors, varianceThreshold, linearityThreshold, normalSimilarityThreshold,
                              distanceThreshold):
    cogs = array(list(map(lambda t: t.reference_point, tensors)))
    cogsBallTree = BallTreePointSet(cogs, leaf_size=20)
    neighbors = cogsBallTree.query(cogs, numNeigbhors + 1)[:, 1:]

    graph = TensorConnectivityGraph(tensors, neighbors, varianceThreshold, normalSimilarityThreshold, distanceThreshold,
                                    linearityThreshold=linearityThreshold)

    # graph.spyGraph()
    nComponents, labels = graph.connected_componnents()
    # graph.connected_segments()

    return labels


if __name__ == '__main__':
    pass
