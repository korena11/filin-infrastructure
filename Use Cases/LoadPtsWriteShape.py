from IOFactory import IOFactory
from PointSet import PointSet
import shapefile

if __name__ == '__main__':
    
    fileName = '..\Sample Data\Geranium2Clouds.pts'
    pointSetList = []
    IOFactory.ReadPts(fileName, pointSetList)
    
    pointSet = pointSetList[0]
    
#     w = shapefile.Writer(shapefile.POINTZ)
#       
#     w.field('intensity', 'N')
#     w.field('red', 'N')
#     w.field('green', 'N')
#     w.field('blue', 'N')
#       
#     map(w.point, pointSet.X(), pointSet.Y(), pointSet.Z())
#     map(w.record, pointSet.Intensity(), pointSet.RGB()[:, 0], pointSet.RGB()[:, 1], pointSet.RGB()[:, 2])
#       
#     w.save('..\\Sample Data\\Shp\\testShape')
#     
#     temp = w.records
#     
#     print temp
    
    IOFactory.WriteToShapeFile(pointSet, '..\\Sample Data\\Shp\\testShape')