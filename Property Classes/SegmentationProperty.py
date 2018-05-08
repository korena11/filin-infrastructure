from numpy import nonzero, random, zeros, uint8

from BaseProperty import BaseProperty
from PointSubSet import PointSubSet


class SegmentationProperty(BaseProperty):
    """
    Holds segments.

    Each segment represented by an integer in range (0, nSegments). 
    The segment -1 is used for unsegmented points
    Can hold result of filter. 

    In this case Count is 2, and segments has values of 0 and 1.
    (Consider inheritance for convenience).

    """
            
    __nSegments = None  # Number of segments. 
    __segments = None  # nX1 ndarray of segmentation labels for each point
    __segmentsColors = None  # # mX3 ndarray of colors for each label (m - number of labels).
    __rgb = None  # nX3 ndarray of colors for each point according to its label
    
    def __init__(self, points, segments):
        """
        Constructor
        
        :param points: reference to points
        :param segments: segmentation labels for each point

        :type points: PointSubSet or PointSet
        :type segments: nx1 nd-array
            
        """
        super(SegmentationProperty, self).__init__(points)

        self.__segments = segments        
        self.__nSegments = len(set(self.__segments))
        
        # Color segments
        # Create a unique color for each segment. Save for future use.
        self.__segmentsColors = 255 * random.random((self.__nSegments, 3))
        # Create rgb array  
        nPoints = points.Size
        self.__rgb = zeros((nPoints, 3), dtype=uint8)
        
        # Assign for each point a color according to the segments it belongs to.                                        
        for i in xrange(0, nPoints):
            self.__rgb[i, :] = self.__segmentsColors[segments[i]]
            
    @property
    def RGB(self):
        return self.__rgb
         
    def GetAllSegments(self):
        
        return self.__segments
    
    def GetSegmentIndices(self, label):
        
        return nonzero(self.__segments == label)[0]        
    
    @property
    def NumberOfSegments(self):
        
        return self.__nSegments
    
    def GetSegment(self, label):
        """
        Return a PointSubSet object from the points in segment labeled "label"
        in case there are no points with the given label return None (and NOT an empty SubSet) 
        
        Args:
            label(int) - the label
            
        Returns:
            PointSubSet object
            
        """
        
        indices = nonzero(self.__segments == label)[0]
        if len(indices) == 0:
            return None
        
        pointSubSet = PointSubSet(self._BaseProperty__points, indices)
        
        return pointSubSet
    
    def UpdatePointLabel(self, pointIndex, newLabel):
        """
        Change the label of a certain point. This method doesn't change the number of labels in the property
        """
        if (pointIndex >= 0 and pointIndex < self._BaseProperty__points.Size and newLabel >= 0 and newLabel < self.__nSegments):
            # Updating label of point
            self.__segments[pointIndex] = newLabel
            self.__rgb[pointIndex, :] = self.__segmentsColors[newLabel]
