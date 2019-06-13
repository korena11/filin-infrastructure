"""
Code for debugging creation of segmentation property based on Zachi's PhD
DO NOT USE FOR OTHER PURPOSES
"""

from IOFactory import IOFactory
from SegmentationFactory import SegmentationFactory, SegmentationProperty
from TensorBallTreeSegmentation import minCutRefinement, dissolveEntrappedSurfaceElements, pointwiseRefinement
from VisualizationO3D import VisualizationO3D

if __name__ == '__main__':
    path = '../../../test_data/'
    # filename = 'test_ply'
    # pntSet = IOFactory.ReadPly(path + filename + '.ply', returnAdditionalAttributes=False)

    # filename = 'test_tensors_2planarSurfaces'

    path = 'C:/Users/zachis/Dropbox/Research/Code/Segmentation/data/'
    # filename = 'agriculture1_clean'
    # filename = 'oldSchool2-clean'
    # filename = 'powerplant5'
    filename = 'tiledWall2'
    pntSet = IOFactory.ReadPts(path + filename + '.pts')

    # segmentation = SegmentationFactory.BallTreeSurfaceElementSegmentation(pntSet, leafSize=10, smallestObjectSize=0.1)

    segmentation2, segmentNeighbors = \
        SegmentationFactory.SurfaceElementsTensorConnectedComponents(pntSet, leafSize=10,
                                                                     smallestObjectSize=0.1, numNeighbors=10,
                                                                     varianceThreshold=0.1 ** 2, linearityThreshold=5,
                                                                     normalSimilarityThreshold=1e-3,
                                                                     distanceThreshold=0.01,  mode='soft_clipping')

    labels, tensors = dissolveEntrappedSurfaceElements(segmentation2, segmentNeighbors,
                                                       varianceThreshold=0.1 ** 2, distanceThreshold=0.05)
    segmentation3 = SegmentationProperty(segmentation2.Points, labels, segmentAttributes=tensors)

    temp = list(map(lambda t: t.tensors_number, tensors))

    labels, tensors = pointwiseRefinement(segmentation3)

    # from numpy import array
    # import open3d as o3d
    # from PointSetOpen3D import PointSetOpen3D
    # tensors = segmentation2.segmentAttributes
    # cogs = array(list(map(lambda t: t.reference_point, tensors)))
    # normals = array(list(map(lambda t: t.plate_axis, tensors)))
    # pcd = PointSetOpen3D(cogs)
    # pcd.data.normals = o3d.Vector3dVector(normals)
    # VisualizationO3D.visualize_pointset(pcd, colors=segmentation2._SegmentationProperty__segmentsColors)

    # temp = minCutRefinement(segmentation2.segmentAttributes, minSegmentSize=3)

    # visObj = VisualizationO3D()
    VisualizationO3D.visualize_pointset(pointset=segmentation2, drawCoordianteFrame=True, coordinateFrameOrigin='min')
    VisualizationO3D.visualize_pointset(pointset=segmentation3, drawCoordianteFrame=True, coordinateFrameOrigin='min')

