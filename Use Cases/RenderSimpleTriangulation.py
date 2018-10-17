from numpy import array

from PointSet import PointSet
from TriangulationFactory import TriangulationFactory
from VisualizationVTK import VisualizationVTK

if __name__ == '__main__':
    
    points = array([[0, -0.5, 0], [1.5, 0, 0], [0, 1, 0], [0, 0, 0.5],
                    [-1, -1.5, 0.1], [0, -1, 0.5], [-1, -0.5, 0],
                    [1, 0.8, 0]], 'f')
    
    pointSet = PointSet(points)
    
    tp = TriangulationFactory.Delaunay2D(pointSet)

    VisualizationVTK.RenderTriangularMesh(tp, 'height', meshRepresentation='surface', colorMap='hot')
    VisualizationVTK.Show()
