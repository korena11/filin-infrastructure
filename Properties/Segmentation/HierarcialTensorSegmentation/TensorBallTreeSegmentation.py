import warnings

from numpy import zeros

from BallTreePointSet import BallTreePointSet
from PointSet import PointSet
from TensorConnectivityGraph import TensorConnectivityGraph
from TensorFactory import TensorFactory


def ExtractSurfaceElements(points, leafSize=10, smallestObjectSize=0.1):
    """
    Partitioning a given point set to surface elements of minimal object size using a ball-tree data-structure
    :param points: point set object (PointSet)
    :param leafSize: smallest number of points allowed in the ball tree data-structure (int)
    :param smallestObjectSize: smallest object expected to be detected in the point set (float)
    :return: a tuple containing:
             - the ball tree of the given point set
             - labels of the points based on the partitioning to nodes
             - the indexes of the nodes in the ball tree that are larger than the given object size by are deepest in
             the ball tree hierarchy levels
             - tensors computed for each of the returned nodes
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

    tensorCompFunc = lambda p: TensorFactory.tensorFromPoints(p, keepPoints=False)
    tensors = list(map(tensorCompFunc, list(map(ballTree.ToNumpy().__getitem__, pointsOfNodes))))

    # Assigning initial labels to the points
    labels = zeros((ballTree.Size,), dtype='i')
    for i in range(len(tensors)):
        labels[pointsOfNodes[i]] = i
    return ballTree, labels, nodesOfInterest, tensors


def tensorConnectedComponents(tensors, numNeighbors, varianceThreshold, linearityThreshold, normalSimilarityThreshold,
                              distanceThreshold, mode='binary'):
    """
    Merging tensors using connected components
    :param tensors: list of tensors (list or ndarray of Tensor objects)
    :param numNeighbors: number of neighbors to find for each tensor (int)
    :param varianceThreshold: max allowed variance of a surface tensor (float)
    :param linearityThreshold: the value between the two largest eigenvalues from which a tensor is considered as
        a linear one (float)
    :param normalSimilarityThreshold:  max difference allowed between the directions of the tensors,
        given in values of the angle sine (float)
    :param distanceThreshold: max distance between two tensors (float)
    :param mode: similarity function indicator. Valid values include: 'binary' (default), 'soft_clipping' and 'exp'
    :return:
    """
    graph = TensorConnectivityGraph(tensors, numNeighbors, varianceThreshold, normalSimilarityThreshold,
                                    distanceThreshold, linearityThreshold=linearityThreshold, mode=mode)

    # graph.spyGraph()
    nComponents, labels, indexesByLabels = graph.connected_componnents()

    return labels, indexesByLabels


if __name__ == '__main__':
    pass
