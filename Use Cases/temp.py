import vtk

from IOFactory import IOFactory
from TriangulationFactory import TriangulationFactory
from VisualizationVTK import VisualizationVTK

if __name__ == '__main__':
    
    pointSetList = []
    fileName = '..\Sample Data\data.xyz'
    IOFactory.ReadXYZ(fileName, pointSetList)
    
    pointSet = pointSetList[0]
    
    tp = TriangulationFactory.Delaunay2D(pointSet)

    VisualizationVTK.RenderTriangularMesh(pointSet, tp, 'height', colorMap='jet')
    
    polyData = pointSet.ToPolyData()
    polyData.polys = tp.TrianglesIndices()
    
    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName('..\\Sample Data\\data.stl')
    
    temp = polyData.deep_copy(polyData)
    
    stlWriter.SetInputConnection(polyData)
    stlWriter.Write()

    VisualizationVTK.Show()
