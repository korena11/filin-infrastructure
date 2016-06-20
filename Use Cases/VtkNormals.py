from IOFactory import IOFactory
from NormalsFactory import NormalsFactory
from PointSet import PointSet


if __name__ == "__main__":
    
    pointSetList = []
    IOFactory.ReadXYZ('..\\Sample Data\\cubeSurface.xyz', pointSetList)
    
    NormalsFactory.VtkNormals(pointSetList[0])
    