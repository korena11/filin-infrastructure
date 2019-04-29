from IOFactory import IOFactory
from Segmentation.SegmentationFactory import SegmentationFactory
from SphericalCoordinatesFactory import SphericalCoordinatesFactory

if __name__ == '__main__':
    
    pointSetList = []
    fileName = '..\\Sample Data\\tigers3.pts'
    IOFactory.ReadPts(fileName, pointSetList)
    
    pointSet = pointSetList[0]
    
    scanLineSegmentation = SegmentationFactory.ScanLinesSegmentation(pointSet)
    
    sphCoord = SphericalCoordinatesFactory.CartesianToSphericalCoordinates(pointSet)
    
    IOFactory.WriteToShapeFile(pointSet, '..\\Sample Data\\Shp\\tigers3', 
                               scanLines = scanLineSegmentation, sphericalCoordinates = sphCoord)
    
    print "Done!"