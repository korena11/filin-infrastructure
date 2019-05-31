"""
Code for debugging creation of segmentation property based on Zachi's PhD
DO NOT USE FOR OTHER PURPOSES
"""

from IOFactory import IOFactory
from SegmentationFactory import SegmentationFactory
from TensorBallTreeSegmentation import tensorConnectedComponents
from VisualizationO3D import VisualizationO3D

if __name__ == '__main__':
    filename = '../../../test_data/test_ply.ply'
    pntSet = IOFactory.ReadPly(filename, returnAdditionalAttributes=False)
    segmentation = SegmentationFactory.BallTreeSurfaceElementSegmentation(pntSet, leafSize=10, smallestObjectSize=0.1)

    tensors = segmentation.getSegmentAttributes
    temp = tensorConnectedComponents(tensors, 10, linearityThreshold=5, varianceThreshold=0.1,
                                     normalSimilarityThreshold=0.9, distanceThreshold=0.1)

    visObj = VisualizationO3D()
    visObj.visualize_pointset(pointset=segmentation, drawCoordianteFrame=True, coordinateFrameOrigin='min')
