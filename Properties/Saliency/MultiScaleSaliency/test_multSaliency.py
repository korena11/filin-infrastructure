from IOmodules.IOFactory import IOFactory
from DataClasses.PointSet import PointSet
from Properties.Segmentation.SegmentationProperty import SegmentationProperty
from VisualizationClasses.VisualizationO3D import VisualizationO3D
from Properties.Curvature.CurvatureProperty import CurvatureProperty
from Properties.Tensors.TensorFactory import TensorFactory

from numpy import arange, int_, vstack, unique, nonzero, array, zeros, ceil, savetxt, hstack
from numpy.linalg import norm
from scipy.sparse import lil_matrix, find
from tqdm import tqdm



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


def getNeighbotingLabels(uniqueCells, labelMapping, label, bufferSize, minNeighborsPerSector=1):
    """
    Getting the neighbors labels of a given cell (defined by its own label)
    :param uniqueCells: list of existing cells given as a list of their respective rows and cols (nx2, ndarray)
    :param labelMapping: grid mapping of the labels of all existing cells (lil_matrix)
    :param label: label of the cell to retrieve its neighbors (int)
    :param bufferSize: the size of the buffer around the center cell to search neighbors in (int)
    :param minNeighborsPerSector: minimum required number of neighbouring cells per sector in order for the sector to
                                  be considered valid (int)
    :return: list of neighboring labels (ndarray) and number of valid sectors (int)
    """
    row, col = uniqueCells[label]  # getting the row and column of the cell based on its label
    neighbors = []  # creating an empty list of neighbors
    validSectors = 0  # counter of the number of valid sectors

    # getting the search boundaries based on the search radius (buffer size) and uniform cells grid size
    minRow = max([0, row - bufferSize])
    minCol = max([0, col - bufferSize])
    maxRow = min([labelMapping.shape[0], row + bufferSize])
    maxCol = min([labelMapping.shape[1], col + bufferSize])

    # TODO: create a private function for finding the neighbors (remove repetitive code)
    if row > 0:
        # finding all the neighbors that are directly below the cell
        rows, cols, labels = find(labelMapping[minRow:row, col])

        if labels.shape[0] > 0:
            neighbors.extend(labels - 1)

        if labels.shape[0] >= minNeighborsPerSector:
            validSectors += 1

    if row > 0 and col > 0:
        # finding all the neighbors that are below and to the left of the cell
        rows, cols, labels = find(labelMapping[minRow:row, minCol:col])

        if labels.shape[0] > 0:
            neighbors.extend(labels - 1)

        if labels.shape[0] >= minNeighborsPerSector:
            validSectors += 1

    if row > 0 and col < labelMapping.shape[1] - 1:
        # finding all the neighbors that are below and to the right of the cell
        rows, cols, labels = find(labelMapping[minRow:row, col + 1:maxCol + 1])

        if labels.shape[0] > 0:
            neighbors.extend(labels - 1)

        if labels.shape[0] >= minNeighborsPerSector:
            validSectors += 1

    if row < labelMapping.shape[0] - 1:
        # finding all the neighbors that are directly above the cell
        rows, cols, labels = find(labelMapping[row + 1:maxRow + 1, col])

        if labels.shape[0] > 0:
            neighbors.extend(labels - 1)

        if labels.shape[0] >= minNeighborsPerSector:
            validSectors += 1

    if row < labelMapping.shape[0] - 1 and col > 0:
        # finding all the neighbors that are above and to the left of the cell
        rows, cols, labels = find(labelMapping[row + 1:maxRow + 1, minCol:col])

        if labels.shape[0] > 0:
            neighbors.extend(labels - 1)

        if labels.shape[0] >= minNeighborsPerSector:
            validSectors += 1

    if row < labelMapping.shape[0] - 1:
        # finding all the neighbors that are above and to the right of the cell
        rows, cols, labels = find(labelMapping[row + 1:maxRow + 1, col + 1:maxCol + 1])

        if labels.shape[0] > 0:
            neighbors.extend(labels - 1)

        if labels.shape[0] >= minNeighborsPerSector:
            validSectors += 1

    if col > 0:
        # finding all the neighbors that are directly to the left of the cell
        rows, cols, labels = find(labelMapping[row, minCol:col])

        if labels.shape[0] > 0:
            neighbors.extend(labels - 1)

        if labels.shape[0] >= minNeighborsPerSector:
            validSectors += 1

    if col < labelMapping.shape[1] - 1:
        # finding all the neighbors that are directly to the right of the cell
        rows, cols, labels = find(labelMapping[row, col + 1:maxCol + 1])

        if labels.shape[0] > 0:
            neighbors.extend(labels - 1)

        if labels.shape[0] >= minNeighborsPerSector:
            validSectors += 1

    # checking if the cell is not considered as its own neighbor
    if label in neighbors:
        raise UserWarning('the cell with label :' + label + ' is found as its own neighbor')

    return array(neighbors, dtype=int), validSectors


def computeUmbrellaCurvaturePerCell(tensors, label, cellNeighbors):
    """
    Computing the umbrella curvature of a cell based on its neighbours
    :param tensors: list of tensors for each cell (list)
    :param label: label of the tensor to compute its curvature (int)
    :param cellNeighbors: the neighboring cells (ndarray of ints)
    :return: the computed umbrella curvature (float)
    """
    # getting the representative points of the neighboring cells
    neighboringPoints = vstack(list(map(lambda n: tensors[n].reference_point, cellNeighbors)))

    # computing the difference vectors of neighboring points with respect to the representative point of the cell
    deltas = neighboringPoints - tensors[label].reference_point

    # normalizing the difference vectors
    normDeltas = norm(deltas, axis=1)
    deltas[:, 0] /= normDeltas
    deltas[:, 1] /= normDeltas
    deltas[:, 2] /= normDeltas

    # computing and returning the umbrella curvature
    return tensors[label].plate_axis.reshape((1, -1)).dot(deltas.T).sum() / neighboringPoints.shape[0]


if __name__ == '__main__':
    # path = 'C:/Zachi/Code/saliency_experiments/ReumaPhD/data/Achziv/'
    # filename = 'Achziv_middle - Cloud_97'

    path = 'C:/Zachi/Code/saliency_experiments/ReumaPhD/data/Tigers/'
    filename = 'tigers - cloud_1M'

    pntSet = IOFactory.ReadPts(path + filename + '.pts')

    cellSize = 0.05
    phenomSize = 0.10
    buffer = int_(ceil(phenomSize / cellSize) / 2)
    buffer = 1 if buffer == 0 else buffer
    print('Neighbor Radius (in num cells): ', buffer)

    rows, cols, labels, uniqueCells, cellLabels = arrangeInUniformCells(pntSet, cellSize)
    numLabels = labels.max() + 1

    segProp = SegmentationProperty(pntSet, labels)

    # VisualizationO3D.visualize_pointset(segProp)

    # smoothSegProp = smoothPointsWithinEachCell(segProp)
    smoothSegProp = segProp

    tensors = list(map(lambda l: TensorFactory.tensorFromPoints(smoothSegProp.GetSegment(l), keepPoints=False),
                       tqdm(range(numLabels), desc='Computing tensors for each cell')))

    neighbors = array(list(map(lambda l: getNeighbotingLabels(uniqueCells, cellLabels, l, buffer),
                           tqdm(range(numLabels), desc='retrieving neighbors for all cells'))))
    validCells = neighbors[:, 1]
    neighbors = neighbors[:, 0]
    validCells = nonzero(validCells >= 7)[0]
    # validCells = nonzero(list(map(lambda l: isValidCell(uniqueCells, cellLabels, l, 7),
    #                               tqdm(range(numLabels), 'checking validity of each cell'))))[0]

    # numNeighbors = list(map(len, neighbors))
    # validCells2 = nonzero(numNeighbors >= ((2 * buffer + 1) ** 2 - 1) * 0.875)[0]

    curvatures = zeros((numLabels, ))
    curvatures[validCells] = list(map(lambda l: computeUmbrellaCurvaturePerCell(tensors, l, neighbors[l]),
                                      tqdm(validCells, desc='computing curvatures for each valid cell')))
    pntCurvatures = zeros((pntSet.Size, ))  # - 999
    list(map(lambda i: pntCurvatures.__setitem__(segProp.GetSegmentIndices(validCells[i]), curvatures[validCells[i]]),
             tqdm(range(validCells.shape[0]), desc='Updating points curvatures')))

    tmp = hstack((pntSet.ToNumpy(), pntCurvatures.reshape((-1, 1))))
    savetxt(path + 'curvature/' + filename + '_uniformCell_' + str(cellSize) + '_' + str(phenomSize) + '.txt',
            tmp, delimiter=',')

    curveProp = CurvatureProperty(smoothSegProp.Points, umbrella_curvature=pntCurvatures)

    visObj = VisualizationO3D()
    visObj.visualize_property(curveProp)

    curvatureMat = zeros(cellLabels.shape) + curvatures.min()
    curvatureMat[uniqueCells[validCells, 0], uniqueCells[validCells, 1]] = curvatures[validCells]

    from matplotlib import pyplot as plt
    plt.imshow(curvatureMat, cmap='jet')
    plt.colorbar()
    plt.show()
