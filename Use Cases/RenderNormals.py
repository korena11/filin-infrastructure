from IOFactory import IOFactory 
from NormalsProperty import NormalsProperty
from NormalsFactory import NormalsFactory
from Visualization import Visualization

if __name__ == "__main__":
    
    pointSetList = []
#    IOFactory.ReadXYZ('..\\Sample Data\\cubeSurface.xyz', pointSetList)
    IOFactory.ReadPts('..\\Sample Data\\Geranium2Clouds.pts', pointSetList)
    
#    normals = NormalsFactory.ReadNormalsFromFile(pointSetList [0], '..\\Sample Data\\cubeSurfaceNormals.xyz')
    normals = NormalsFactory.VtkNormals(pointSetList[0])
    
    Visualization.RenderPointSet(normals, 'color', color=(0, 0, 0), pointSize = 3.0)
    Visualization.Show()
        
    
    