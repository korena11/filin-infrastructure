"""
Code for debugging creation of segmentation property based on Zachi's PhD
DO NOT USE FOR OTHER PURPOSES
"""

from IOFactory import IOFactory
from SegmentationFactory import SegmentationFactory
from VisualizationO3D import VisualizationO3D

if __name__ == '__main__':
    filename = '../../../test_data/test_ply.ply'
    pntSet = IOFactory.ReadPly(filename, returnAdditionalAttributes=False)
    segmentation = SegmentationFactory.BallTreeSurfaceElementSegmentation(pntSet, leafSize=10, smallestObjectSize=0.1)

    visObj = VisualizationO3D()
    visObj.visualize_pointset(pointset=segmentation, drawCoordianteFrame=True, coordinateFrameOrigin='min')
