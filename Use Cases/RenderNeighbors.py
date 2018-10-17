from random import randint

from IOFactory import IOFactory
from NeighborsFactory import NeighborsFactory
from PointSubSet import PointSubSet
from VisualizationVTK import VisualizationVTK

if __name__ == "__main__":
    
    fileName = '..\Sample Data\Geranium2Clouds.pts'
    pointSetList = []
    IOFactory.ReadPts(fileName, pointSetList)
    
    points = pointSetList[0]
    
    i = randint(0, points.Size())

    searchPointSubSet = PointSubSet(points, [i])
    pointsInRange = NeighborsFactory.GetNeighborsIn3dRange(points, points.X()[i], points.Y()[i], points.Z()[i], 0.02)

    _figure = VisualizationVTK.RenderPointSet(points, 'rgb')
    _figure = VisualizationVTK.RenderPointSet(searchPointSubSet, 'color', pointSize=5.0, color=(1, 0, 0),
                                              _figure=_figure)
    VisualizationVTK.RenderPointSet(pointsInRange, 'color', pointSize=2.0, color=(1, 0, 1), _figure=_figure)
    VisualizationVTK.Show()
