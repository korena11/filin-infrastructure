"""
Code for debugging creation of segmentation property based on Zachi's PhD
DO NOT USE FOR OTHER PURPOSES
"""

from IOFactory import IOFactory
from SegmentationFactory import SegmentationFactory, SegmentationProperty
from TensorBallTreeSegmentation import tensorConnectedComponents
from VisualizationO3D import VisualizationO3D

if __name__ == '__main__':
    path = '../../../test_data/'
    # filename = 'test_ply.ply'
    # pntSet = IOFactory.ReadPly(path + filename, returnAdditionalAttributes=False)

    filename = 'test_tensors_2planarSurfaces.pts'

    path = 'C:/Users/zachis/Dropbox/Research/Code/Segmentation/data/'
    # filename = 'agriculture1_clean.pts'
    filename = 'oldSchool2.pts'
    pntSet = IOFactory.ReadPts(path + filename)

    segmentation = SegmentationFactory.BallTreeSurfaceElementSegmentation(pntSet, leafSize=10, smallestObjectSize=0.1)

    tensors = segmentation.getSegmentAttributes
    temp = tensorConnectedComponents(tensors, 10, linearityThreshold=5, varianceThreshold=0.1 ** 2,
                                     normalSimilarityThreshold=0.001, distanceThreshold=0.01, mode='soft_clipping')

    segmentation2 = SegmentationProperty(pntSet, temp[segmentation.GetAllSegments])

    # visObj = VisualizationO3D()
    VisualizationO3D.visualize_pointset(pointset=segmentation2, drawCoordianteFrame=True, coordinateFrameOrigin='min')
