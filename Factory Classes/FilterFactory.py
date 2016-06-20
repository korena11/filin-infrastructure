from numba import autojit
from numpy import sqrt, arctan, ones, uint8, pi, min, max, logical_and, nonzero
from IOFactory import IOFactory
from Visualization import Visualization
from SegmentationProperty import SegmentationProperty 

@autojit
def SlopeBasedMorphologicFilter(pntData, searchRadius, slopeThreshold):
    numPoints = len(pntData)
    groundPointsIndices = []
    
    slopeThreshold = slopeThreshold  * pi / 180
    
    for i in xrange(numPoints):
        isGround = True
        for j in xrange(numPoints):
            dist = sqrt((pntData[i, 0] - pntData[j, 0]) ** 2 + (pntData[i, 1] - pntData[j, 1]) ** 2)
            if (dist < searchRadius and pntData[i, 2] > pntData[j, 2]):
                slope = arctan((pntData[i, 2] - pntData[j, 2]) / dist)
                if (slope > slopeThreshold):
                    isGround = False                        
                    break
        
        if (isGround):
            groundPointsIndices.append(i)
    
    return groundPointsIndices

class FilterFactory:       
    """
    Filter PointSet using different methods
    """
    @staticmethod        
    def __CreateSegmentationProperty(points, indices):
        numPoints = points.Size()
        segments = ones((numPoints), dtype=uint8)
        segments[indices] = 0
        
        return SegmentationProperty(points, segments) 
        
    @staticmethod        
    def SlopeBasedMorphologicFilter(points, searchRadius, slopeThreshold):
        """
        Slope Based Morphological Filter
        
        :Args:
        -----
            - points (PointSet)
            - searchRadius (float): search radius in meters
            - slopeThreshold (float): maximum slope angle allowed, given in degrees 
            
        :Returns:
        ------- 
            - SegmentationProperty
            
            segment 0 contains the terrain points
        
        """
        pntData = points.ToNumpy()
               
        groundPointsIndices = SlopeBasedMorphologicFilter(pntData, searchRadius, slopeThreshold)

        # 0 - terrain, 1 - cover
        return FilterFactory.__CreateSegmentationProperty(points, groundPointsIndices)        

    @staticmethod
    def FilterByBoundingBox(points, xmin = None, xmax = None, ymin = None, ymax = None, zmin = None, zmax = None):
        """
        Filtering based on a defined boundary box
        
        :Args:
            - points (PointSet)
            - xmin, xmax, ymin, ymax, zmin, zmax 
            
        :Returns:
            - SegmentationProperty
            
            segment 0 contains the non-filtered points
        
        """
        
        # Defining values of unsent parameters
        if (xmin == None):
            xmin = min(points.X())
            
        if (xmax == None):
            xmax = max(points.X())
            
        if (ymin == None):
            ymin = min(points.Y())
            
        if (ymax == None):
            ymax = max(points.Y())
            
        if (zmin == None):
            zmin = min(points.Z())
            
        if (zmax == None):
            zmax = max(points.Z())
            
        # Finding the indices of all the points inside the defined bounds
        insidePointIndices = nonzero(logical_and(logical_and(logical_and(points.X() >= xmin, points.X() <= xmax),
                                                              logical_and(points.Y() >= ymin, points.Y() <= ymax)),
                                                  logical_and(points.Z() >= zmin, points.Z() <= zmax)))
        
        return FilterFactory.__CreateSegmentationProperty(points, insidePointIndices)
    
    @staticmethod
    def FilterBySphericalCoordinates(sphCoorProp, minAzimuth = 0, maxAzimuth = 360,
                                     minElevationAngle = -90, maxElevationAngle = 90, minRange = 0, maxRange = None):
        """
        Filtering based on spherical coordinates values
        """
        
        if (sphCoorProp == None):
            return None
            
        # If maxRange was not provided, defining as the highest range
        if (maxRange == None):
            maxRange = max(sphCoorProp.Ranges())
        
        # Finding the indices of all the points inside the defined bounds
        insidePointIndices = nonzero(logical_and(logical_and(logical_and(sphCoorProp.Azimuths() >= minAzimuth, 
                                                                          sphCoorProp.Azimuths() <= maxAzimuth),
                                                              logical_and(sphCoorProp.ElevationAngles() >= minElevationAngle, 
                                                                          sphCoorProp.ElevationAngles() <= maxElevationAngle)),
                                                  logical_and(sphCoorProp.Ranges() >= minRange, 
                                                              sphCoorProp.Ranges() <= maxRange)))

        return FilterFactory.__CreateSegmentationProperty(sphCoorProp.Points(), insidePointIndices)
            
if __name__ == '__main__':
    
    pointSetList = []
    IOFactory.ReadXYZ('..\\Sample Data\\DU9_2.xyz', pointSetList)

#    filterFactory = FilterFactory()
    terrainSubSet = FilterFactory.SlopeBasedMorphologicFilter(pointSetList[0], 1.0, 25).GetSegment(0)
    
    Visualization.RenderPointSet(terrainSubSet, 'color', color=(0.5, 0, 0))
    Visualization.Show()