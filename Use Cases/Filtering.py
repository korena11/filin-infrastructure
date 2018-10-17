from numpy import min, max

from FilterFactory import FilterFactory
from IOFactory import IOFactory
from VisualizationVTK import VisualizationVTK

if __name__ == '__main__':
    
    pointSetList = []
    fileName = '..\\Sample Data\\tigers3.pts'
    IOFactory.ReadPts(fileName, pointSetList)
    
    pointSet = pointSetList[0]
    
#    t1 = datetime.now()
#    
#    morphologicFilterResults = FilterFactory.SlopeBasedMorphologicFilter(pointSet, 0.5, 15)
#    
#    t2 - datetime.now()
#    
#    print t2 - t1
#    
#    terrainSubSet = morphologicFilterResults.GetSegment(0)
#    Visualization.RenderPointSet(terrainSubSet, 'color', color=(0.5, 0, 0))
    
    boxFilterResults = FilterFactory.FilterByBoundingBox(pointSet, min(pointSet.X()) + 1.0, max(pointSet.X()) - 1.0, 
                                                         min(pointSet.Y()) + 1.0, max(pointSet.Y()) - 1.0, 
                                                         min(pointSet.Z()) + 0.01, max(pointSet.Z()) - 0.01)
    
    boxSubSet = boxFilterResults.GetSegment(0)

    VisualizationVTK.RenderPointSet(boxSubSet, 'color', color=(0.5, 0, 0))
    
    sphFilterResults =  FilterFactory.FilterBySphericalCoordinates(pointSet, None, 180, 250, -40, -10, 2, 7.5)
    
    sphSubSet = sphFilterResults.GetSegment(0)

    VisualizationVTK.RenderPointSet(sphSubSet, 'color', color=(0.5, 0, 0))

    VisualizationVTK.Show()
