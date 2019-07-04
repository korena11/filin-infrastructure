import warnings

from numpy import zeros, array, nonzero, triu_indices, unique, int_, hstack
from numpy.linalg import norm

from scipy.sparse import coo_matrix

from tqdm import tqdm
from networkx import DiGraph, minimum_cut
from networkx.exception import NetworkXUnbounded

from BallTreePointSet import BallTreePointSet
from PointSet import PointSet
from TensorConnectivityGraph import TensorConnectivityGraph
from TensorFactory import TensorFactory
from TensorSet import TensorSet

from SegmentMinCutter import SegmentMinCutter


def extractSurfaceElements(points, leafSize=10, smallestObjectSize=0.1):
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


def dissolveEntrappedSurfaceElements(segmentation, segmentNeighbors=None, dominantSegmentSize=10, minSegmentSize=5,
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
        raise ValueError('Segments neighbors are missing')

    labels = segmentation.GetAllSegments
    tensors = segmentation.segmentAttributes

    # getting the number of tensors composing each segment
    segmentSizes = array(list(map(lambda t: t.tensors_number, tensors)))

    smallSegments = nonzero(segmentSizes <= minSegmentSize)[0]  # extracting small segments
    smallNeighbors = segmentNeighbors[smallSegments]  # extracting the neighbors of the small segments
    neighborSizes = list(map(segmentSizes.__getitem__, smallNeighbors))  # getting the sizes of the neighboring segments

    # identifying the dominant segments around each small ones
    dominantNeighbors = list(map(lambda neighbors, sizes: neighbors[sizes >= dominantSegmentSize],
                                 smallNeighbors, neighborSizes))
    numDominantNeighbors = array(list(map(len, dominantNeighbors)))  # getting the number of dominant neighbors

    # identifying small segments which have either a single dominant neighbor or a pair of dominant ones
    entrappedBySingle = nonzero(numDominantNeighbors == 1)[0]
    entrappedByPair = nonzero(numDominantNeighbors == 2)[0]

    if len(entrappedBySingle) > 0 or len(entrappedByPair) > 0:

        # dissolving small segments entrapped by a single dominant segment
        for i in tqdm(entrappedBySingle, 'Dissolving surface elements entrapped between a single dominant segment'):
            dominantNeighbor = dominantNeighbors[i][0]  # getting the index of the dominant neighbor

            # skipping segments with high variance
            if tensors[smallSegments[i]].eigenvalues[0] > varianceThreshold:
                continue

            # skipping segments whose center of gravity is relatively far from the tensor
            # TODO: redefine in terms of variance of larger segment
            if tensors[dominantNeighbor].distanceFromPoint(
                    tensors[smallSegments[i]].reference_point) > distanceThreshold:
                continue

            # TODO: change to adding the tensors of the smaller segment and not the overall one
            tensors[dominantNeighbor].addTensor(tensors[smallSegments[i]])  # merging the tensors
            tensors[smallSegments[i]] = None  # nullifying the tensor of the smaller segment
            labels[labels == smallSegments[i]] = dominantNeighbor  # reassigning the points of the small segment

        # dissolving small segments entrapped by a pair of dominant segments
        for i in tqdm(entrappedByPair, 'Dissolving surface elements entrapped between a pair of dominant segments'):
            segmentPoints = segmentation.GetSegmentIndices(smallSegments[i])
            dominantNeighbor1 = dominantNeighbors[i][0]
            dominantNeighbor2 = dominantNeighbors[i][1]

            distancesFrom1 = tensors[dominantNeighbor1].distanceFromPoint(
                segmentation.Points.ToNumpy()[segmentPoints]).reshape((-1, ))
            distancesFrom2 = tensors[dominantNeighbor2].distanceFromPoint(
                segmentation.Points.ToNumpy()[segmentPoints]).reshape((-1, ))

            graph = DiGraph()
            list(map(lambda p, d: graph.add_edge('source', p, capacity=d ** -2), segmentPoints, distancesFrom1))
            list(map(lambda p, d: graph.add_edge(p, 'sink', capacity=d ** -2), segmentPoints, distancesFrom2))

            indexes1, indexes2 = triu_indices(segmentPoints.shape[0], 1)
            indexes1 = segmentPoints[indexes1]
            indexes2 = segmentPoints[indexes2]
            pointDistances = norm(segmentation.Points.ToNumpy()[indexes1] -
                                  segmentation.Points.ToNumpy()[indexes2], axis=1)
            list(map(lambda pi, pj, d: graph.add_edge(pi, pj, capacity=d ** -2), indexes1, indexes2, pointDistances))
            list(map(lambda pi, pj, d: graph.add_edge(pj, pi, capacity=d ** -2), indexes1, indexes2, pointDistances))

            try:
                cutValue, partition = minimum_cut(graph, 'source', 'sink')
                reachable, notReachable = partition

                reachable = array(list(reachable))
                notReachable = array(list(notReachable))

                if 'source' in reachable and 'sink' in notReachable:
                    intReachable = int_(reachable[reachable != 'source'])
                    intNotReachable = int_(notReachable[notReachable != 'sink'])

                    if len(intReachable) > 0:
                        newTensor1 = TensorFactory.tensorFromPoints(segmentation.Points.ToNumpy()[intReachable])
                        tensors[dominantNeighbor1].addTensor(newTensor1)
                        labels[intReachable] = dominantNeighbor1

                    if len(intNotReachable) > 0:
                        newTensor2 = TensorFactory.tensorFromPoints(segmentation.Points.ToNumpy()[intNotReachable])
                        tensors[dominantNeighbor2].addTensor(newTensor2)
                        labels[intNotReachable] = dominantNeighbor2

                    tensors[smallSegments[i]] = None

            except NetworkXUnbounded:
                continue
            # except LinAlgError:
            #     continue

        # removing tensors of dissolved segments from the list
        tensors = tensors[nonzero(tensors)[0]]

        # updating points labels
        uniqueLabels = unique(labels)
        uniqueNewLabels = range(uniqueLabels.shape[0])
        newOldCovesion = dict(zip(uniqueLabels, uniqueNewLabels))
        newLabels = array(list(map(lambda l: newOldCovesion[l], labels)))

        return newLabels, tensors

    else:
        warnings.warn('No surface elements that are entrapped by a single segment were found')
        return labels, tensors

def pointwiseRefinement(segmentation, significantSegmentSize=10, maxIterations=1000):
    """

    :param segmentation:
    :param significantSegmentSize:
    :return:
    """
    if not isinstance(segmentation.segmentAttributes[0], TensorSet):
        raise TypeError('Segmentation attributes are not TensorSets objects')

    from EnergyBasedSegmentRefiner import EnergyBasedSegmentRefiner

    refiner = EnergyBasedSegmentRefiner(segmentation, significantSegmentSize)

    fails = 0
    i = 0
    print(refiner.energy)
    while i < maxIterations:
        if refiner.optimizeEnergy():
            fails = 0
            i += 1
            print(refiner.energy)
        else:
            fails += 1
            if fails > 10:
                break

    return





if __name__ == '__main__':
    pass
