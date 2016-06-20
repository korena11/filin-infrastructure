from numpy import fabs, sum, mean, median, sqrt, tile, nonzero, asarray, logical_or, logical_not, ones, roll
from PointSet import PointSet
from BaseProperty import BaseProperty

class TriangulationProperty(BaseProperty):
    '''
    TriangulationProperty - Class for holding and analyzing the triangulation data of a given point set
    '''

    def __init__(self, points, trianglesIndices):
        '''
        Constructor:
        @param points: A PointSet Object
        @param trianglesIndices: a n-by-3 array holding the indices, each row representing a triangle 
        '''
        
        self._BaseProperty__points = points
        self.__trianglesIndices = trianglesIndices
        self.__numTriangles = len(trianglesIndices)
    
    @property        
    def NumberOfTriangles(self):
        '''
        Returns the number of triangles
        @return: The number of triangles
        '''
        return self.__numTriangles
    
    @property    
    def TrianglesIndices(self):
        '''
        Returns the indices of all the triangles
        '''
        return self.__trianglesIndices  
    
      
    def AreaOfTriangle(self, index):
        '''
        Calculating the area of a specific triangle defined by its index in the triangles list
        @param index: The index of the triangle
        @return: The area of the triangle 
        '''
        if (index < 0 or index >= self.__numTriangles):
            return 0
        else:
            trianglePoints = self._BaseProperty__points.ToNumpy()[self.__trianglesIndices[index]]
            return 0.5 * fabs(trianglePoints[0, 0] * (trianglePoints[1, 1] - trianglePoints[2, 1]) + 
                              trianglePoints[1, 0] * (trianglePoints[2, 1] - trianglePoints[0, 1]) + 
                              trianglePoints[2, 0] * (trianglePoints[0, 1] - trianglePoints[1, 1]))
           
           
    def TotalArea(self):
        '''
        Calculating the total area enclosed in the triangulation
        @return: The total area of all the triangles in the triangulation
        '''
        return sum(map(self.AreaOfTriangle, xrange(self.__numTriangles)))
      
      
    def AverageTriangleArea(self):
        '''
        Calculating the average area of all triangles
        @return: The average area of all the triangles
        '''
        return mean(map(self.AreaOfTriangle, xrange(self.__numTriangles)))
    
      
    def MedianTriangleArea(self):
        '''
        Calculating the median area of all triangles
        @return: The median area of all the triangles
        '''
        return median(map(self.AreaOfTriangle, xrange(self.__numTriangles)))
    
    
    def LengthOfEdge(self, triangleIndex, edgeIndex):
        '''
        Calculating the length of a specific edge of one of the triangles
        @param triangleIndex: The index of a triangle
        @param edgeIndex: The index of the edge (0 - the edge between the first two points of the triangle,
                                                 1 - the edge between the last two points of the triangle,
                                                 2 - the edge between the first point and the last point)
        '''
        if ((triangleIndex < 0 or triangleIndex >= self.__numTriangles) or (edgeIndex < 0 or edgeIndex >= 3)):
            return 0
        else:
            trianglePoints = self._BaseProperty__points.ToNumpy()[self.__trianglesIndices[triangleIndex]]
            
            if (edgeIndex == 0):
                return sqrt((trianglePoints[0, 0] - trianglePoints[1, 0]) ** 2 + 
                            (trianglePoints[0, 1] - trianglePoints[1, 1]) ** 2 + 
                            (trianglePoints[0, 2] - trianglePoints[1, 2]) ** 2)
            elif (edgeIndex == 1):
                return sqrt((trianglePoints[1, 0] - trianglePoints[2, 0]) ** 2 + 
                            (trianglePoints[1, 1] - trianglePoints[2, 1]) ** 2 + 
                            (trianglePoints[1, 2] - trianglePoints[2, 2]) ** 2)
            else:
                return sqrt((trianglePoints[0, 0] - trianglePoints[2, 0]) ** 2 + 
                            (trianglePoints[0, 1] - trianglePoints[2, 1]) ** 2 + 
                            (trianglePoints[0, 2] - trianglePoints[2, 2]) ** 2)
    
  
    def LengthOfAllEdges(self):
        '''
        Calculating edges' length in the triangulation
        '''
        from numpy import asarray, hstack, expand_dims
        return hstack((expand_dims(asarray(map(self.LengthOfEdge, xrange(self.__numTriangles), tile(0, self.__numTriangles))), 1),
                expand_dims(asarray(map(self.LengthOfEdge, xrange(self.__numTriangles), tile(1, self.__numTriangles))), 1),
                expand_dims(asarray(map(self.LengthOfEdge, xrange(self.__numTriangles), tile(2, self.__numTriangles))), 1)))
    
   
    def AverageEdgeLength(self):
        '''
        Calculating the average edge length in the triangulation
        '''
        edgeLength = self.LengthOfAllEdges()
        return (mean(edgeLength[0]) + mean(edgeLength[1]) + mean(edgeLength[2])) / 3.0

                     
    def MedianEdgeLength(self):
        '''
        Calculating the median edge length in the triangulation
        '''
        edgeLengths = map(self.LengthOfEdge, xrange(self.__numTriangles), tile(0, self.__numTriangles))
        edgeLengths.extend(map(self.LengthOfEdge, xrange(self.__numTriangles), tile(1, self.__numTriangles)))
        edgeLengths.extend(map(self.LengthOfEdge, xrange(self.__numTriangles), tile(2, self.__numTriangles)))
        return median(edgeLengths)
    
   
    def TrimEdgesByLength(self, maxLength):
        '''
        Removing all triangles which have an edge longer than a given threshold
        @return: Number of triangles removed
        '''
        currentNumOfTriangles = self.__numTriangles
        
        edges1 = asarray(map(self.LengthOfEdge, xrange(self.__numTriangles), tile(0, self.__numTriangles)))
        edges2 = asarray(map(self.LengthOfEdge, xrange(self.__numTriangles), tile(1, self.__numTriangles)))
        edges3 = asarray(map(self.LengthOfEdge, xrange(self.__numTriangles), tile(2, self.__numTriangles))) 
        trianglesToKeep = nonzero(logical_not(logical_or(logical_or(edges1 > maxLength,
                                                                      edges2 > maxLength),
                                               edges3 > maxLength)))[0]
        
        self.__trianglesIndices = self.__trianglesIndices[trianglesToKeep]
        self.__numTriangles = len(trianglesToKeep)
        
        return currentNumOfTriangles - self.__numTriangles

            
    def IsPointInTriangle(self, x, y, triangleIndex):
        '''
        Finding if a given point (x, y) is inside one of the triangles (defined by its index)
        @return: 1 - if the point is inside the triangle, 0 - if the point lies on one or more of the triangle edges, -1 - the point is outside the triangle
        '''
        if (triangleIndex < 0 or triangleIndex >= self.__numTriangles):
            return -1
        
        trianglePoints = self._BaseProperty__points.ToNumpy()[self.__trianglesIndices[triangleIndex]]
        rolledTrianglePoints = roll(trianglePoints, -1, axis=0)
        
        # Calculating perpendicular distances of the point from all of the triangle edges
        perpendicularDistances = ((rolledTrianglePoints[:, 0] - trianglePoints[:, 0]) * (y - trianglePoints[:, 1]) - 
                                  (rolledTrianglePoints[:, 1] - trianglePoints[:, 1]) * (x - trianglePoints[:, 0]))
                
        if (len(perpendicularDistances[perpendicularDistances > 0]) == 3 or 
            len(perpendicularDistances[perpendicularDistances < 0]) == 3):
            # If all distances are positive or all distances are negative, the point is inside the triangle
            return 1
        elif (len(perpendicularDistances[perpendicularDistances == 0]) > 0):
            # If one or more of the perpendicular distances is zero, the point is on the triangle boundary
            return 0
        else:
            return -1
    
if __name__ == "__main__":
    
    from numpy import array
    
    points = array([[0, -0.5, 0], [1.5, 0, 0], [0, 1, 0], [0, 0, 0.5],
                    [-1, -1.5, 0.1], [0, -1, 0.5], [-1, -0.5, 0],
                    [1, 0.8, 0]], 'f')
    
    triangles = array([(0, 1, 3), (1, 2, 3), (1, 0, 5), (2, 3, 4), (3, 0, 4), (0, 5, 4), (2, 4, 6), (2, 1, 7)])
    
    pointSet = PointSet(points)
    tp = TriangulationProperty(pointSet, triangles)
    
    print map(tp.IsPointInTriangle, 0.5 * ones((tp.NumberOfTriangles, 1)), 0.5 * ones((tp.NumberOfTriangles, 1)), xrange(tp.NumberOfTriangles))
    
    print "Number of points:", tp.Points.Size
    print "Number of triangles:", tp.NumberOfTriangles
    print "Area of first triangle:", tp.AreaOfTriangle(0)
    print "Total area of all triangles:", tp.TotalArea()
    print "Average area of triangles:", tp.AverageTriangleArea()
    print "Median area of triangles:", tp.MedianTriangleArea()
    print "Length of the first edge of the first triangle: ", tp.LengthOfEdge(0, 0)
    print "Average length of edges: ", tp.AverageEdgeLength()
    print "Median length of edges: ", tp.MedianEdgeLength()
    print "Removing triangles with edges larger than ", 2.0
    print "Number of triangles removed: ", tp.TrimEdgesByLength(2.0)
