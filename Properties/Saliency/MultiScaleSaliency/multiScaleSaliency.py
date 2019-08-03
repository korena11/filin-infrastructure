from IOFactory import IOFactory
from PointSet import PointSet
from SegmentationFactory import SegmentationProperty
from VisualizationO3D import VisualizationO3D

from numpy import arange, int_, vstack, unique
from numpy.linalg import norm
from scipy.sparse import lil_matrix


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
    return cols, rows, labels, cellLabels


def smoothPointsWithinEachCell(segProp):
    """
    Smoothing heights of points within each cell (corresponding to segment)
    :param segProp: Segmentated point set where each segments (SegmentationPropoerty)
    :return: A smoothed segmented point set (SegmentationProperty)
    """
    numLabels = segProp.NumberOfSegments  # getting the number of labels

    # computing mean height within each cell
    meanHeightPerCell = list(map(lambda l: segProp.Points.ToNumpy()[segProp.GetSegmentIndices(l), 2].mean(),
                                 range(numLabels)))
    pnts = segProp.Points.ToNumpy()
    newHeights = pnts[:, 2]  # getting original heights
    list(map(lambda l: newHeights.__setitem__(segProp.GetSegmentIndices(l), meanHeightPerCell[l]), range(numLabels)))
    pnts[:, 2] = newHeights  # updating heights

    # recreating segmentation property with smoothed points
    return SegmentationProperty(PointSet(pnts), segProp.GetAllSegments)


def computeUmbrellaCurvaturePerCell(cellPoints, cellTensor):
    """

    :param cellPoints:
    :param cellTensor:
    :return:
    """
    if cellPoints.Size < 8:
        return 0.0
    else:
        deltas = cellPoints.ToNumpy() - cellTensor.reference_point
        normDeltas = norm(deltas, axis=1)
        deltas[:, 0] /= normDeltas
        deltas[:, 1] /= normDeltas
        deltas[:, 2] /= normDeltas
        return cellTensor.plate_axis.reshape((1, -1)).dot(deltas.T).sum()


if __name__ == '__main__':
    path = 'C:/Zachi/Code/saliency_experiments/ReumaPhD/data/Achziv/'
    filename = 'Achziv_middle - Cloud_97'
    pntSet = IOFactory.ReadPts(path + filename + '.pts')

    cellSize = 0.50

    rows, cols, labels, cellLabels = arrangeInUniformCells(pntSet, cellSize)
    numLabels = labels.max() + 1

    segProp = smoothPointsWithinEachCell(SegmentationProperty(pntSet, labels))

    from TensorFactory import TensorFactory
    tensors = list(map(lambda l: TensorFactory.tensorFromPoints(segProp.GetSegment(l)), range(numLabels)))

    computeUmbrellaCurvaturePerCell(segProp.GetSegment(0), tensors[0])

    VisualizationO3D.visualize_pointset(segProp)
