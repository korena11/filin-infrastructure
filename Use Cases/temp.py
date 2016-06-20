from IOFactory import IOFactory
from PointSet import PointSet
from TriangulationFactory import TriangulationFactory
from Visualization import Visualization
import vtk

if __name__ == '__main__':
    
    pointSetList = []
    fileName = '..\Sample Data\data.xyz'
    IOFactory.ReadXYZ(fileName, pointSetList)
    
    pointSet = pointSetList[0]
    
    tp = TriangulationFactory.Delaunay2D(pointSet)
    
    Visualization.RenderTriangularMesh(pointSet, tp, 'height', colorMap = 'jet')
    
    polyData = pointSet.ToPolyData()
    polyData.polys = tp.TrianglesIndices()
    
    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName('..\\Sample Data\\data.stl')
    
    temp = polyData.deep_copy(polyData)
    
    stlWriter.SetInputConnection(polyData)
    stlWriter.Write()
    
    Visualization.Show()