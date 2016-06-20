from PointSet import PointSet
from SegmentationFactory import SegmentationFactory 
from Visualization import Visualization
from IOFactory import IOFactory
    
if __name__ == '__main__':

    pointSetList = []
    fileName = '..\\Sample Data\\tigers3.pts'
    IOFactory.ReadPts(fileName, pointSetList)
        
    scanLineSegmentation = SegmentationFactory.ScanLinesSegmentation(pointSetList[0])
        
    Visualization.RenderPointSet(scanLineSegmentation, 'segmentation')
    Visualization.Show()
    