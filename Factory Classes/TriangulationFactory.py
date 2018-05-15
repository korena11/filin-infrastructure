from mayavi import mlab
from numpy import array

from PointSet import PointSet
from TriangulationProperty import TriangulationProperty


class TriangulationFactory:
    '''
    Create triangulation for a set of points using different methods 
    '''
    @staticmethod
    def Delaunay2D( points ):
        '''
        Creating a 2D Delaunay triangulation for a given set of points

        :param points:

        :type points: PointSet

        :returns: TriangulationProperty object

        '''
        fig2Close = mlab.figure( bgcolor = ( 0.5, 0.5, 0.5 ), fgcolor = ( 1, 1, 1 ) )
        
        # Getting the points from the PointSet object in vtkPolyData format
        polyData = points.ToPolyData()
        
        # Running the 2D Delaunay triangulation
        delaunay = mlab.pipeline.delaunay2d( polyData ).outputs[0]

        # Extracting the triangles indices from the vtkPolyData
        triangles = delaunay.polys.data.to_array()
        trianglesIndices = triangles.reshape( len( triangles ) / 4, 4 )[:, 1: 4]
        
        mlab.close()  # Closing the figure created from the running the delaunay triangulation
        
        # Returning a TriangulationProperty Object
        return TriangulationProperty( points, trianglesIndices )
        
if __name__ == '__main__':
    
    points = array( [[0, -0.5, 0], [1.5, 0, 0], [0, 1, 0], [0, 0, 0.5],
                    [-1, -1.5, 0.1], [0, -1, 0.5], [-1, -0.5, 0],
                    [1, 0.8, 0]], 'f' )
    
    pointSet = PointSet( points )
    
    tp = TriangulationFactory.Delaunay2D( pointSet )
    
    print tp.NumberOfTriangles()
