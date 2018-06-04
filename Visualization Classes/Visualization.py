# from mayavi import mlab
# import matplotlib.pyplot as plt
# from mayavi import mlab
# from mayavi.core import lut_manager
# from mayavi.mlab import quiver3d
from numpy import tile, asarray, expand_dims, uint8
from scipy.ndimage import morphology as morph

import SphericalCoordinatesProperty
from BaseProperty import BaseProperty
from ColorProperty import ColorProperty
from NormalsProperty import NormalsProperty
from PointSet import PointSet
from PointSubSet import PointSubSet
from SegmentationProperty import SegmentationProperty
from TriangulationFactory import TriangulationFactory
from TriangulationProperty import TriangulationProperty


class Visualization:
    
    @classmethod
    def __CreatePolyDataForRendering( cls, points, renderFlag, color = ( 0, 0, 0 ) ):
        '''
        
        '''
        if ( isinstance( points, BaseProperty ) ):
            polyData = points.Points.ToPolyData()
            numPoints = points.Points.Size
        else:
            polyData = points.ToPolyData()
            numPoints = points.Size
                        
        if renderFlag == 'color':
            polyData.point_data.scalars = asarray( 255 * tile( color, ( numPoints, 1 ) ), dtype = uint8 )        
        elif renderFlag == 'externRgb' and isinstance( points, ColorProperty ):
            polyData.point_data.scalars = points.RGB          
        elif renderFlag == 'rgb' and ( isinstance( points, PointSet ) or isinstance( points, PointSubSet ) ) and points.RGB != None:            
                polyData.point_data.scalars = points.RGB    
        elif renderFlag == 'intensity' and ( isinstance( points, PointSet ) or isinstance( points, PointSubSet ) ) and points.Intensity != None:
            polyData.point_data.scalars = points.Intensity            
        elif renderFlag == 'height' and ( isinstance( points, PointSet ) or isinstance( points, PointSubSet ) ) and points.Z != None:
            polyData.point_data.scalars = points.Z
        elif renderFlag == 'height' and isinstance( points, BaseProperty ):
            polyData.point_data.scalars = points.Points.Z
        elif renderFlag == 'segmentation' and isinstance( points, SegmentationProperty ):
            polyData.point_data.scalars = points.RGB     
        elif renderFlag == 'parametericColor':
            if len( color[0].shape ) == 1:
                param = expand_dims( color[0], 1 )
            else:
                param = color[0]
            polyData.point_data.scalars = param              
        else:  # display in some default color
            print('rendering using default color') 
            polyData.point_data.scalars = asarray( 255 * tile( ( 0.5, 0, 0 ), ( len( polyData.points ), 1 ) ), dtype = uint8 )
            polyData.point_data.scalars.name = 'default'
            
        polyData.point_data.scalars.name = renderFlag            
        
        return polyData
            
    @classmethod    
    def RenderPointSet( cls, points, renderFlag, _figure = None, color = ( 0, 0, 0 ), pointSize = 1.0, colorMap = None, colorBar = 0 ):
        """
        Render a PointSet object as points
        Can use/render the following attributes
            rgb color of each point 
            intensity of each point
            Normal for each point                 
        
        Args:
            points (PointSet/PointSubSet):
            
            renderFlag (str): how to render the points. Can be one of:
                'rgb': color the points using the original rgb values
                'intensity': color the points using the original intensity values
                'segmentation': color the points using segmentation results and given colormap
                'height': color the points using the z coordinate 
                'color': render all points using this color
                'parametericColor': render accoding to given parameter
            
            color (3-tuple): color used to render points in case renderFlag = 'color'     
            
            _figure (mlab.figure, optional): Mayavi scene
            
            Optional Property objects:
            
            segmentationProperty (SegmentationProperty, optional)            
            normalsProperty (NormalsProperty, optional)     
            colorProperty
            
            pointSize (float, optional): size of the points when rendered
        """                        
        if ( points == None or ( ( isinstance( points, PointSet ) or isinstance( points, PointSubSet ) ) and points.Size == 0 ) or
            ( isinstance( points, BaseProperty ) and points.Points.Size == 0 ) ):
            return None  
        
        # Create a new figure
        if _figure == None: 
            _figure = mlab.figure( bgcolor = ( 0.8, 0.8, 0.8 ), fgcolor = ( 1, 1, 1 ) )
                      
        polyData = cls.__CreatePolyDataForRendering( points, renderFlag, color )
        
        if ( colorMap != None and colorMap in lut_manager.lut_mode_list() ):
            s = mlab.pipeline.surface( polyData, colormap = colorMap )
        else:
            s = mlab.pipeline.surface( polyData )

        s.actor.property.set( point_size = pointSize, representation = 'points' )
        if colorBar == 1:
            mlab.colorbar( orientation = 'horizontal', nb_labels = 5, label_fmt = '%.3f' )
        
        if isinstance( points, NormalsProperty ):
            quiver3d( points.Points.X, points.Points.Y, points.Points.Z, points.dX, points.dY, points.dZ, scale_factor = 0.1, colormap = 'hsv' )
        
        return _figure        
    
    @classmethod
    def RenderTriangularMesh( cls, points, renderFlag, _figure = None, color = ( 0, 0, 0 ), renderProperty = None, meshRepresentation = 'wireframe', colorMap = None ):
        '''
        
        '''
        if ( points == None or ( ( isinstance( points, PointSet ) or isinstance( points, PointSubSet ) ) and points.Size == 0 ) or
            ( isinstance( points, BaseProperty ) and points.Points.Size == 0 ) ):
            return None  
        
        if ( not isinstance( points, TriangulationProperty ) ):
            points = TriangulationFactory.Delaunay2D( points )
        
        if ( isinstance( points.Points, SphericalCoordinatesProperty.SphericalCoordinatesProperty ) ):
            points.Points = points.Points.XYZ
        
        # Create a new figure
        if ( _figure == None ): 
            _figure = mlab.figure( bgcolor = ( 0.5, 0.5, 0.5 ), fgcolor = ( 1, 1, 1 ) )
        
        if ( renderProperty == None ):
            polyData = cls.__CreatePolyDataForRendering( points.Points, renderFlag, color )
        else:
            polyData = cls.__CreatePolyDataForRendering( renderProperty.Points, renderFlag, color )
            
        polyData.polys = points.TrianglesIndices
        
        if ( colorMap != None and colorMap in lut_manager.lut_mode_list() ):
            s = mlab.pipeline.surface( polyData, colormap = colorMap )
        else:
            s = mlab.pipeline.surface( polyData )
        
        s.actor.property.set( representation = meshRepresentation, line_width = 1.0 )
        
        return _figure
    
    @classmethod
    def ShowPanorama( cls, panorama, colormap = None, _figure = None ):
        """
        Rendering a panorama image. In order to view the panorama use pyplot.show()
        
        :Args:
            - panorama - A PanoramaProperty object
            - colorMap - The color map used for rendering (optional)
            - _figure - The figure to show the panorama on (optional)
        """
        if ( _figure == None ):
            _figure = plt.figure()
        
        pano = morph.grey_erosion( uint8( panorama.PanoramaImage ), size = ( 3, 3 ) )
#         pano = panorama.PanoramaImage
        
        plt.imshow( pano, colormap )
        plt.axis( 'off' )
        return _figure
    
    @staticmethod
    def Show():
        mlab.show()
