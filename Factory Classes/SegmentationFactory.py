import datetime
from random import randint

from numpy import fabs, nonzero, asarray, int, ones, arange

from SegmentationProperty import SegmentationProperty
from SphericalCoordinatesFactory import SphericalCoordinatesFactory


class SegmentationFactory:
    """
    Create different types of segmentations

    The results are saved to SegmentationProperty object
    """

#     @classmethod    
#     def ScanLinesSegmentation(cls, points):
#         """
#         For now, we read the segmentation results from pickled file
#         
#         Args:
#             points (pointSet)
#          
#         """
#         # read pickled file
#         fileName = '..\\Sample Data\\tigers3ScanLines.p'
#         f = open(fileName , "r")
#         segments = pickle.load(f)                
#         f.close()
#         
#         segmentationProperty = SegmentationProperty(points, segments)
#         return segmentationProperty
    
    @classmethod
    def __ScanLineRansac(cls, azimuths, elevationAngles, inlierThr):
        numPnts = len(azimuths)
    
        # Initializing RANSAC Parameters
        initialNumIterations = 1000
        numIterations = initialNumIterations
        numParams = 3
    #    inlierThr = scanResolutionAz * 0.025;
        maxNumInliers = 0;
        bestParams = []
        
        i = 0
        while (i < numIterations):
            i += 1
            index1 = randint(0, numPnts - 1)
            index2 = index1
            index3 = index1
        
            while (index2 == index1):
                index2 = randint(0, numPnts - 1)
            
            while (index3 == index1 or index3 == index2):
                index3 = randint(0, numPnts - 1)
        
            x1 = elevationAngles[index1]
            x2 = elevationAngles[index2]
            x3 = elevationAngles[index3]
            
            y1 = azimuths[index1]
            y2 = azimuths[index2]
            y3 = azimuths[index3]
            
            a1 = ((y2 - y1) * (x3 ** 2 - x1 ** 2) - (y3 - y1) * (x2 ** 2 - x1 ** 2)) / ((x2 - x1) * (x3 ** 2 - x1 ** 2) - (x3 - x1) * (x2 ** 2 - x1 ** 2))
            a2 = (y2 - y1) / (x2 ** 2 - x1 ** 2) - 1.0 / (x2 + x1) * a1
            a0 = y1 - a2 * x1 ** 2 - a1 * x1
            
            diffs = fabs(a0 + a1 * elevationAngles + a2 * elevationAngles ** 2 - azimuths)
            
            inliersIndexes = nonzero(diffs < inlierThr)[0]
            numInliers = len(inliersIndexes)
            
            if (numInliers > maxNumInliers):
                maxNumInliers = numInliers
                maxInliersIndexes = inliersIndexes
                bestParams = [a0 , a1, a2]
                numIterations = int((1.0 / numInliers) ** numParams + 0.1 * initialNumIterations)
        
        return maxInliersIndexes, asarray(bestParams)
    
    @classmethod
    def ParabolicScanLineSegmentation(cls, pointSet, scanResolutionAz, searchRatio=0.75, ransacRatio=0.025, minimalScanLineLength=500):
        
        sphCoorProp = SphericalCoordinatesFactory.CartesianToSphericalCoordinates(pointSet)
        
        pntIndexes = arange(pointSet.Size)
        scanLineIndex = -1 * ones((pointSet.Size), dtype=int)
        currentScanLineIndex = -1
        pointList = []
        paramsList = []
        
        now = datetime.datetime.now()
        print((datetime.date(now.year, now.month, now.day), datetime.time(now.hour, now.minute, now.second),
               "- Starting Phase 1 of Sorting Process"))
        startTime = now
        
        azimuths = sphCoorProp.Azimuths
        elevationAngles = sphCoorProp.ElevationAngles
        
        while (len(pntIndexes) > 0):

            print(("Progress Status:", float(int((1 - float(len(pntIndexes)) / pointSet.Size) * 10000000)) / 100000,
                   "% complete"))
            
            startAzimuth = azimuths[pntIndexes[0]]
            
            tmpIndexes = (nonzero(fabs(azimuths[pntIndexes] - startAzimuth) < searchRatio * scanResolutionAz))[0]
            
            numPnts = len(tmpIndexes)
            
            if (numPnts < minimalScanLineLength):
                pntIndexes[tmpIndexes] = -10
                pntIndexes = pntIndexes[pntIndexes != -10]
                continue
            
            currentScanLineIndex += 1
            
            tmpAzimuths = azimuths[pntIndexes[tmpIndexes]]
            tmpElevationAngles = elevationAngles[pntIndexes[tmpIndexes]]
            maxInliersIndexes, params = cls.__ScanLineRansac(tmpAzimuths, tmpElevationAngles, scanResolutionAz * ransacRatio)
            
            scanLineIndex[pntIndexes[tmpIndexes[maxInliersIndexes]]] = currentScanLineIndex
            pointList.append(pntIndexes[tmpIndexes[maxInliersIndexes]])
            paramsList.append(params)
            
            pntIndexes[tmpIndexes[maxInliersIndexes]] = -10
            pntIndexes = pntIndexes[pntIndexes != -10]
        
        now = datetime.datetime.now()
        print((datetime.date(now.year, now.month, now.day), datetime.time(now.hour, now.minute, now.second),
               "- End of Phase 1 - Total time:", now - startTime))
        print(("            Number of scan lines:", currentScanLineIndex))
        print(("            Number of unsorted points: ", len(scanLineIndex[scanLineIndex == -1]), "out of",
               pointSet.Size))
        
        unsortedIndexes = nonzero(scanLineIndex == -1)[0]
        
        now = datetime.datetime.now()
        print((datetime.date(now.year, now.month, now.day), datetime.time(now.hour, now.minute, now.second),
               "- Starting Phase 2 of Sorting Process"))
        
        closestLineIndex = -1 * ones((len(unsortedIndexes)), dtype=int)
        minDiffs = 100 * ones((len(unsortedIndexes)))
        
        for i in range(currentScanLineIndex):
            
            diffs = fabs(paramsList[i][0] + paramsList[i][1] * elevationAngles[unsortedIndexes] + paramsList[i][2] * 
                            elevationAngles[unsortedIndexes] ** 2 - azimuths[unsortedIndexes])
            
            closestLineIndex[minDiffs > diffs] = i
            minDiffs[minDiffs > diffs] = diffs[minDiffs > diffs]
        
        now = datetime.datetime.now()
        print((datetime.date(now.year, now.month, now.day), datetime.time(now.hour, now.minute, now.second),
               "- End of Phase 2"))
        
        now = datetime.datetime.now()
        print((datetime.date(now.year, now.month, now.day), datetime.time(now.hour, now.minute, now.second),
               "- Creating SegmentationProperty Object"))
        
        segmentationProperty = SegmentationProperty(pointSet, scanLineIndex)
        list(map(segmentationProperty.UpdatePointLabel, unsortedIndexes, closestLineIndex))
        
        return segmentationProperty
        
if __name__ == '__main__':
    
    from IOFactory import IOFactory
    from VisualizationVTK import VisualizationVTK
    from numpy import random
       
    pointSetList = []
    fileName = 'D:\\Documents\\Pointsets\\set3_1.pts' 
    IOFactory.ReadPts(fileName, pointSetList)
    
    scanLineSegmentation = SegmentationFactory.ParabolicScanLineSegmentation(pointSetList[0], 0.1, minimalScanLineLength=5)
        
    colors = 255 * random.random((scanLineSegmentation.NumberOfSegments, 3))
    print (colors)
    
    _figure = None
    for i in range(scanLineSegmentation.NumberOfSegments):
        scanLinei = scanLineSegmentation.GetSegment(i)
        _figure = VisualizationVTK.RenderPointSet(scanLinei, 'color', color=colors[i], _figure=_figure, pointSize=3)
    VisualizationVTK.Show()
