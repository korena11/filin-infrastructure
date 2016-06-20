# In this example:
# 1. Create PointSet objects from .pts file
# 2. Render the PointSets as as points using one color for each set
from PointSet import PointSet
from Visualization import Visualization
from IOFactory import IOFactory

if __name__ == "__main__":
    
    fileName = '..\Sample Data\Geranium2Clouds.pts'
    pointSetList = []
    IOFactory.ReadPts(fileName, pointSetList)
    
    _figure = None
    colors = [(0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9)]
    for color, pointSet in zip(colors, pointSetList):
        _figure = Visualization.RenderPointSet(pointSet, 'color', _figure, color=color)
        
#    for color, pointSet in zip(colors, pointSetList):
#        _figure = Visualization.RenderPointSet(pointSet, 'rgb', color=color)
        
    Visualization.Show()
    
