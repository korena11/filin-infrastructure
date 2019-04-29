from IOFactory import IOFactory
from Segmentation.SegmentationFactory import SegmentationFactory
from VisualizationVTK import VisualizationVTK

if __name__ == '__main__':

    pointSetList = []
    fileName = '..\\Sample Data\\tigers3.pts'
    IOFactory.ReadPts(fileName, pointSetList)
        
    scanLineSegmentation = SegmentationFactory.ScanLinesSegmentation(pointSetList[0])

    VisualizationVTK.RenderPointSet(scanLineSegmentation, 'segmentation')
    VisualizationVTK.Show()
