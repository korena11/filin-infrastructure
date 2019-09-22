"""
Code for debugging creation of segmentation property based on Zachi's PhD
DO NOT USE FOR OTHER PURPOSES
"""

from IOmodules.IOFactory import IOFactory
from Properties.Segmentation.SegmentationFactory import SegmentationFactory   #, SegmentationProperty
from Properties.Segmentation.SegmentationProperty import SegmentationProperty
from TensorBallTreeSegmentation import minCutRefinement, dissolveEntrappedSurfaceElements, pointwiseRefinement
from VisualizationClasses.VisualizationO3D import VisualizationO3D

if __name__ == '__main__':
    path = '../../../test_data/'
    # filename = 'test_ply'
    # pntSet = IOFactory.ReadPly(path + filename + '.ply', returnAdditionalAttributes=False)

    filename = 'test_tensors_2planarSurfaces'

    path = 'C:/Users/zachis/Dropbox/Research/Code/Segmentation/data/'
    # filename = 'agriculture1_clean'
    filename = 'oldSchool2-clean'
    # filename = 'powerplant5'
    # filename = 'curvedWall'
    # filename = 'waterTower2'
    pntSet = IOFactory.ReadPts(path + filename + '.pts')

    # # segmentation = SegmentationFactory.BallTreeSurfaceElementSegmentation(pntSet,
    # #                                                                       leafSize=10, smallestObjectSize=0.1)

    # segmentation2, segmentNeighbors = \
    #     SegmentationFactory.SurfaceElementsTensorConnectedComponents(pntSet, leafSize=10,
    #                                                                  smallestObjectSize=0.1, numNeighbors=10,
    #                                                                  varianceThreshold=0.05 ** 2, linearityThreshold=5,
    #                                                                  normalSimilarityThreshold=0.01,
    #                                                                  distanceThreshold=0.08,  mode='binary')
    #
    # labels, tensors, neighbors = minCutRefinement(segmentation2, segmentNeighbors)
    # segmentation3 = SegmentationProperty(segmentation2.Points, labels, segmentAttributes=tensors)
    #
    # labels, tensors, neighbors = dissolveEntrappedSurfaceElements(segmentation3, neighbors,
    #                                                               varianceThreshold=0.05 ** 2,
    #                                                               distanceThreshold=0.08,
    #                                                               minSegmentSize=3)
    #
    # #
    # # temp = list(map(lambda t: t.tensors_number, tensors))
    # #
    from pickle import dump, load
    # dump((labels, tensors, neighbors), open(filename + '_seg.p', 'wb'))
    labels, tensors, neighbors = load(open(filename + '_seg.p', 'rb'))

    segmentation4 = SegmentationProperty(pntSet, labels, segmentAttributes=tensors)
    labels, tensors = pointwiseRefinement(segmentation4, neighbors)

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
    # VisualizationO3D.visualize_pointset(pointset=segmentation2, drawCoordianteFrame=True, coordinateFrameOrigin='min')
    # VisualizationO3D.visualize_pointset(pointset=segmentation3, drawCoordianteFrame=True, coordinateFrameOrigin='min')
    VisualizationO3D.visualize_pointset(pointset=segmentation4, drawCoordianteFrame=True, coordinateFrameOrigin='min')

