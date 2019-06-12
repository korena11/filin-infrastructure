import warnings

from numpy import zeros, array, nonzero, triu_indices, unique, logical_and

from BallTreePointSet import BallTreePointSet
from PointSet import PointSet
from TensorConnectivityGraph import TensorConnectivityGraph
from TensorFactory import TensorFactory
from TensorSet import TensorSet

from SegmentMinCutter import SegmentMinCutter


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

    nComponents, labels, indexesByLabels = graph.connected_componnents()

    segmentNeighbors = graph.collapseConnectivity()

    return labels, indexesByLabels, segmentNeighbors


def minCutRefinement(tensors, dominantSegmentSize=10, minSegmentSize=3, numNeighbors=10):
    """
    WIP - DO NOT USE!!!
    :param tensors:
    :param minSegmentSize:
    :param numNeighbors:
    :return:
    """
    # if isinstance(tensors, list):
    #     tensors = array(tensors)
    segmentSizes = array(list(map(lambda t: t.tensors_number, tensors)))

    dominantSegments = nonzero(segmentSizes > dominantSegmentSize)[0]

    # cogs = array(list(map(lambda t: t.reference_point, tensors[dominantSegments])))
    # bt = BallTreePointSet(cogs, leaf_size=10)
    # neighbors = bt.query(cogs, numNeighbors)[:, 1:]

    indexes1, indexes2 = triu_indices(dominantSegments.shape[0], 1)
    newSegmentLabels = range(len(tensors))

    for i, j in zip(indexes1, indexes2):
        if newSegmentLabels[dominantSegments[i]] != dominantSegments[i] or \
                newSegmentLabels[dominantSegments[j]] != dominantSegments[j]:
            continue  # one of the segments has been merged

        smc = SegmentMinCutter(tensors[dominantSegments[i]], tensors[dominantSegments[j]])
        if smc.minCut():
            # merging all the tensors from both segments into one
            list(map(tensors[dominantSegments[i]].addTensor, tensors[dominantSegments[j]].tensors))
            newSegmentLabels[dominantSegments[j]] == dominantSegments[i]
            print(dominantSegments[i], dominantSegments[j])

    uniqueNewLabels = unique(newSegmentLabels)
    newLabels = range(uniqueNewLabels.shape[0])
    # newSegmentLabels = newLabels[]


def dissolveEntrappedSurfaceElements(segmentation, segmentNeighbors=None, dominantSegmentSize=10, minSegmentSize=3,
                                     varianceThreshold=0.01, distanceThreshold=0.1):
    """

    :param segmentationProperty:
    :param segmentNeighbors:
    :return:
    """
    if not isinstance(segmentation.segmentAttributes[0], TensorSet):
        raise TypeError('Segmentation attributes are not TensorSets objects')

    if segmentNeighbors is None:
        # TODO: replace with neighbor querying
        raise AttributeError('Segments neighbors are missing')

    labels = segmentation.GetAllSegments
    tensors = segmentation.segmentAttributes
    segmentSizes = array(list(map(lambda t: t.tensors_number, tensors)))

    smallSegments = nonzero(segmentSizes < minSegmentSize)[0]
    smallNeighbors = segmentNeighbors[smallSegments]
    neighborSizes = list(map(segmentSizes.__getitem__, smallNeighbors))

    dominantNeighbors = list(map(lambda neighbors, sizes: neighbors[sizes >= dominantSegmentSize],
                                 smallNeighbors, neighborSizes))
    numDominantNeighbors = array(list(map(len, dominantNeighbors)))

    # singleNeighbors = nonzero(list(map(lambda n: len(n) == 1, smallNeighbors)))[0]
    # singleNeighborsSize = segmentSizes[list(map(lambda i: smallNeighbors[i][0], singleNeighbors))]

    entrappedBySingle = nonzero(numDominantNeighbors == 1)[0]
    entrappedByDuo = nonzero(numDominantNeighbors == 2)[0]

    if len(entrappedBySingle) > 0 or len(entrappedByDuo) > 0:

        for i in entrappedBySingle:
            dominantNeighbor = dominantNeighbors[i][numDominantNeighbors[i].argmax()]

            if tensors[smallSegments[i]].eigenvalues[0] > varianceThreshold:
                continue

            if tensors[dominantNeighbor].distanceFromPoint(
                    tensors[smallSegments[i]].reference_point) > distanceThreshold:
                continue

            tensors[dominantNeighbor].addTensor(tensors[smallSegments[i]])
            tensors[smallSegments[i]] = None
            labels[labels == smallSegments[i]] = dominantNeighbor

        tensors = tensors[nonzero(tensors)[0]]

        uniqueLabels = unique(labels)
        uniqueNewLabels = range(uniqueLabels.shape[0])
        newOldCovesion = dict(zip(uniqueLabels, uniqueNewLabels))
        newLabels = array(list(map(lambda l: newOldCovesion[l], labels)))

        return newLabels, tensors

    else:
        warnings.warn('No surface elements that are entrapeed by a single segment were found')
        return labels, tensors


if __name__ == '__main__':
    pass
