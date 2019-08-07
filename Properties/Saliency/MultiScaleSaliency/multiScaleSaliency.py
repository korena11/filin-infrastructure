from IOFactory import IOFactory
from PointSet import PointSet
from SegmentationFactory import SegmentationProperty
from VisualizationO3D import VisualizationO3D
from CurvatureProperty import CurvatureProperty

from numpy import arange, int_, vstack, unique, nonzero, array, zeros
from numpy.linalg import norm
from scipy.sparse import lil_matrix
from tqdm import tqdm

import cProfile


def arrangeInUniformCells(pntSet, cellSize):
    """
    Arranging the point into a uniform cells data structure
    :param pntSet: point set to arrange into uniform cells (PointSet)
    :param cellSize: size of cells (float)
    :return:
    """

    # getting lower left corner coordinates
    xmin = pntSet.ToNumpy()[:, 0].min()
    ymin = pntSet.ToNumpy()[:, 1].min()

    # computing cell indexes of each point in the point cloud
    cols = int_((pntSet.ToNumpy()[:, 0] - xmin) / cellSize)
    rows = int_((pntSet.ToNumpy()[:, 1] - ymin) / cellSize)

    # getting list of unique cell ids
    cellIds = vstack([rows, cols]).T
    uniqueCellIds = unique(cellIds, axis=0)

    # adding a label to each cell
    cellLabels = lil_matrix((rows.max() + 1, cols.max() + 1))
    cellLabels[uniqueCellIds[:, 0], uniqueCellIds[:, 1]] = arange(uniqueCellIds.shape[0]) + 1

    # applying labels to points
    labels = int_(list(map(lambda r, c: cellLabels[r, c] - 1, rows, cols)))
    return cols, rows, labels, uniqueCellIds, cellLabels


def smoothPointsWithinEachCell(segProp):
    """
    Smoothing heights of points within each cell (corresponding to segment)
    :param segProp: Segmentated point set where each segments (SegmentationPropoerty)
    :return: A smoothed segmented point set (SegmentationProperty)
    """
    numLabels = segProp.NumberOfSegments  # getting the number of labels

    # computing mean height within each cell
    meanHeightPerCell = list(map(lambda l: segProp.Points.ToNumpy()[segProp.GetSegmentIndices(l), 2].mean(),
                                 tqdm(range(numLabels), desc='Computing mean height for each cell')))
    pnts = segProp.Points.ToNumpy()
    newHeights = pnts[:, 2]  # getting original heights
    list(map(lambda l: newHeights.__setitem__(segProp.GetSegmentIndices(l), meanHeightPerCell[l]),
             tqdm(range(numLabels), desc='Updating points heights')))
    pnts[:, 2] = newHeights  # updating heights

    # recreating segmentation property with smoothed points
    return SegmentationProperty(PointSet(pnts), segProp.GetAllSegments)


def isValidCell(uniqueCells, labelMapping, label, minNumNeighbors=7):
    """

    :param uniqueCells: list of existing cells given as a list of their respective rows and cols (nx2, ndarray)
    :param labelMapping: grid mapping of the labels of all existing cells (lil_matrix)
    :param label: label of the cell to check
    :param minNumNeighbors: minimum number of neighboring cells required for cell to valid
    :return: True if cell has a minimum required number of neighbors, otherwise False
    """
    return getNeighbotingLabels(uniqueCells, labelMapping, label).shape[0] >= minNumNeighbors


def getNeighbotingLabels(uniqueCells, labelMapping, label):
    """
    Getting the neighbors labels of a given cell (defined by its own label)
    :param uniqueCells: list of existing cells given as a list of their respective rows and cols (nx2, ndarray)
    :param labelMapping: grid mapping of the labels of all existing cells (lil_matrix)
    :param label: label of the cell to retrieve its neighbors
    :return: list of neighboring labels (ndarray)
    """
    row, col = uniqueCells[label]
    neighbors = []
    if row > 0 and labelMapping[row - 1, col] > 0:
        neighbors.append(labelMapping[row - 1, col] - 1)

    if row > 0 and col > 0 and labelMapping[row - 1, col - 1] > 0:
        neighbors.append(labelMapping[row - 1, col - 1] - 1)

    if row > 0 and col < labelMapping.shape[1] - 1 and labelMapping[row - 1, col + 1] > 0:
        neighbors.append(labelMapping[row - 1, col + 1] - 1)

    if row < labelMapping.shape[0] - 1 and labelMapping[row + 1, col] > 0:
        neighbors.append(labelMapping[row + 1, col] - 1)

    if row < labelMapping.shape[0] - 1and col > 0 and labelMapping[row + 1, col - 1] > 0:
        neighbors.append(labelMapping[row + 1, col - 1] - 1)

    if row < labelMapping.shape[0] - 1 and col < labelMapping.shape[1] - 1 and labelMapping[row + 1, col + 1] > 0:
        neighbors.append(labelMapping[row + 1, col + 1] - 1)

    if col > 0 and labelMapping[row, col - 1] > 0:
        neighbors.append(labelMapping[row, col - 1] - 1)

    if col < labelMapping.shape[1] - 1 and labelMapping[row, col + 1] > 0:
        neighbors.append(labelMapping[row, col + 1] - 1)

    return array(neighbors)


def computeUmbrellaCurvaturePerCell(segProp, tensor, cellNeighbors):
    """

    :param segProp:
    :param tensor:
    :return:
    """
    neighboringPoints = vstack(list(map(lambda n: segProp.GetSegment(n).ToNumpy(), cellNeighbors)))

    deltas = neighboringPoints - tensor.reference_point
    normDeltas = norm(deltas, axis=1)
    deltas[:, 0] /= normDeltas
    deltas[:, 1] /= normDeltas
    deltas[:, 2] /= normDeltas

    return tensor.plate_axis.reshape((1, -1)).dot(deltas.T).sum()


if __name__ == '__main__':
    path = 'C:/Zachi/Code/saliency_experiments/ReumaPhD/data/Achziv/'
    filename = 'Achziv_middle - Cloud_97'
    pntSet = IOFactory.ReadPts(path + filename + '.pts')

    cellSize = 0.1

    rows, cols, labels, uniqueCells, cellLabels = arrangeInUniformCells(pntSet, cellSize)
    numLabels = labels.max() + 1

    segProp = SegmentationProperty(pntSet, labels)

    VisualizationO3D.visualize_pointset(segProp)

    smoothSegProp = smoothPointsWithinEachCell(segProp)

    from TensorFactory import TensorFactory
    tensors = list(map(lambda l: TensorFactory.tensorFromPoints(smoothSegProp.GetSegment(l)),
                       tqdm(range(numLabels), desc='Computing tensors for each cell')))

    neighbors = list(map(lambda l: getNeighbotingLabels(uniqueCells, cellLabels, l),
                         tqdm(range(numLabels), desc='retrieving neighbors for all cells')))
    validCells = nonzero(list(map(lambda n: n.shape[0] >= 7, neighbors)))[0]
    # validCells = nonzero(list(map(lambda l: isValidCell(uniqueCells, cellLabels, l, 7),
    #                               tqdm(range(numLabels), 'checking validity of each cell'))))[0]

    curvatures = zeros((numLabels, ))
    curvatures[validCells] = list(map(lambda l: computeUmbrellaCurvaturePerCell(smoothSegProp,
                                                                                tensors[l], neighbors[l]),
                                      tqdm(validCells, desc='computing curvatures for each valid cell')))
    pntCurvatures = zeros((pntSet.Size, ))
    list(map(lambda i: pntCurvatures.__setitem__(segProp.GetSegmentIndices(validCells[i]), curvatures[validCells[i]]),
             tqdm(range(validCells.shape[0]), desc='Updating points curvatures')))

    curveProp = CurvatureProperty(smoothSegProp.Points, umbrella_curvature=curvatures)


    visObj = VisualizationO3D()
    visObj.visualize_property(curveProp)

    curvatureMat = zeros(cellLabels.shape) + curvatures.min()
    curvatureMat[uniqueCells[validCells, 0], uniqueCells[validCells, 1]] = curvatures[validCells]

    from matplotlib import pyplot as plt
    plt.imshow(curvatureMat, cmap='jet')
    plt.colorbar()
    plt.show()
