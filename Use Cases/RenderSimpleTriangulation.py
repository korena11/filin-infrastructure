from Visualization import Visualization
from numpy import array
from PointSet import PointSet
from TriangulationFactory import TriangulationFactory

if __name__ == '__main__':
    
    points = array([[0, -0.5, 0], [1.5, 0, 0], [0, 1, 0], [0, 0, 0.5],
                    [-1, -1.5, 0.1], [0, -1, 0.5], [-1, -0.5, 0],
                    [1, 0.8, 0]], 'f')
    
    pointSet = PointSet(points)
    
    tp = TriangulationFactory.Delaunay2D(pointSet)
    
    Visualization.RenderTriangularMesh(tp, 'height', meshRepresentation = 'surface', colorMap = 'hot')
    Visualization.Show()