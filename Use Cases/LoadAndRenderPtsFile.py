# In this example:
# 1. Create PointSet objects from .pts file
# 2. Render the PointSets as as points using one color for each set
from IOFactory import IOFactory
from VisualizationVTK import VisualizationVTK

if __name__ == "__main__":
    
    fileName = '/home/photo-lab-3/Dropbox/PhD/Data/ActiveContours/PCLs/channel2.pts'
    pointSetList = []
    IOFactory.ReadPts(fileName, pointSetList)
    
    _figure = None
    colors = [(0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9)]
    for color, pointSet in zip(colors, pointSetList):
        _figure = VisualizationVTK.RenderPointSet(pointSet, 'color', _figure, color=color)
        
#    for color, pointSet in zip(colors, pointSetList):
#        _figure = Visualization.RenderPointSet(pointSet, 'rgb', color=color)

    VisualizationVTK.Show()
