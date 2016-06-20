from Visualization import Visualization
from IOFactory import IOFactory
from PointSet import PointSet
from TriangulationFactory import TriangulationFactory

if __name__ == '__main__':
    
    filename = '..\Sample Data\data.xyz'
    pointSetList = []
    IOFactory.ReadXYZ(filename, pointSetList)
    
    pointSet = pointSetList[0]
    
    tp = TriangulationFactory.Delaunay2D(pointSet)
    
    Visualization.RenderTriangularMesh(tp, 'height', meshRepresentation = 'surface', colorMap = 'jet')
    
    print "Number of triangles:", tp.NumberOfTriangles()
    print "Area of first triangle:", tp.AreaOfTriangle(0)
    print "Total area of all triangles:", tp.TotalArea()
    print "Average area of triangles:", tp.AverageTriangleArea()
    print "Median area of triangles:", tp.MedianTriangleArea()
    print "Length of the first edge of the first triangle: ", tp.LengthOfEdge(0, 0)
    print "Average length of edges: ", tp.AverageEdgeLength()
    print "Median length of edges: ", tp.MedianEdgeLength()
    
    maxLength = 10.0
    
    print "Removing triangles with edges larger than ", maxLength
    print "Number of triangles removed: ", tp.TrimEdgesByLength(maxLength)
    
    Visualization.RenderTriangularMesh(tp, 'height', meshRepresentation = 'surface', colorMap = 'jet')
    
    Visualization.Show()