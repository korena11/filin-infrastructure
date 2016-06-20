from BaseProperty import BaseProperty
from SphericalCoordinatesProperty import SphericalCoordinatesProperty
from SphericalCoordinatesFactory import SphericalCoordinatesFactory
from PanoramaProperty import PanoramaProperty
from ColorProperty import ColorProperty
from SegmentationProperty import SegmentationProperty
from NormalsProperty import NormalsProperty 
from numpy import min, max, int_

class PanoramaFactory:
    """
    Creates a panoramic view from point set based on a certain property (e.g. range, intesity, etc.)
    The panoramic view is stored as a PanoramProperty object
    """
    
    @classmethod
    def CreatePanorama( cls, points, _property = 'range', azimuthSpacing = 0.057, elevationSpacing = 0.057 ):
        """
        Creating a PanoramProperty object from a point set based on certain propery
        
        :Args:
            - points - The point set to create the panorama from can be either a PointSet, PointSubSet or any other BaseProperty-derived objects
            - _property - The property to create the panorama according to. Can be: 'range', 'intensity', 'color', 'segmentation', 'normals'
            - azimuthSpacing - The spacing between two points of the point set in the azimuth direction (scan property)
            - elevationSpacing - The spacing between two points of the point set in the elevation direction (scan property)
        """
        if ( isinstance( points, SphericalCoordinatesProperty ) ):
            sphCoords = points
        else:
            # Calculating the spherical coordinates of the point set
            sphCoords = SphericalCoordinatesFactory.CartesianToSphericalCoordinates( points )
            
        # Retrieving the original point set
        if ( isinstance( points, BaseProperty ) ):
            pointSet = points.Points
        else:
            pointSet = points
            
        # Finding the boundaries of the panorama
        minAz = min( sphCoords.Azimuths )
        maxAz = max( sphCoords.Azimuths )
        minEl = min( sphCoords.ElevationAngles )
        maxEl = max( sphCoords.ElevationAngles )
        
        # Calculating the location of each point in the panorama
        azimuthIndexes = int_( ( sphCoords.Azimuths - minAz ) / azimuthSpacing )
        elevationIndexes = int_( ( maxEl - sphCoords.ElevationAngles ) / elevationSpacing )
        
        # Creating the panorams based on the requested property
        if ( _property == 'range' ):  # Creating a range panorama
            return PanoramaProperty( pointSet, elevationIndexes, azimuthIndexes, sphCoords.Ranges, dataType = _property, minAzimuth = minAz, maxAzimuth = maxAz,
                                    minElevation = minEl, maxElevation = maxEl, azimuthSpacing = azimuthSpacing, elevationSpacing = elevationSpacing )
        
        elif ( _property == 'intensity' ):  # Creating an inensity panorama
            return PanoramaProperty( pointSet, elevationIndexes, azimuthIndexes, pointSet.Intensity, dataType = _property, minAzimuth = minAz, maxAzimuth = maxAz,
                                    minElevation = minEl, maxElevation = maxEl, azimuthSpacing = azimuthSpacing, elevationSpacing = elevationSpacing )
        
        elif ( _property == 'color' ):  # Creating a panorama based on the colors of point set
            
            # Checking if the input 'points' object is an instance of either ColorProperty or SegmenationProperty which have color data
            if ( isinstance( points, ColorProperty ) or isinstance( points, SegmentationProperty ) ): 
                return PanoramaProperty( pointSet, elevationIndexes, azimuthIndexes, points.RGB, dataType = _property, minAzimuth = minAz, maxAzimuth = maxAz,
                                    minElevation = minEl, maxElevation = maxEl, azimuthSpacing = azimuthSpacing, elevationSpacing = elevationSpacing )
            
            # Checking if the original PointSet has color data in it 
            elif ( pointSet.RGB != None ):
                return PanoramaProperty( pointSet, elevationIndexes, azimuthIndexes, pointSet.RGB, dataType = _property, minAzimuth = minAz, maxAzimuth = maxAz,
                                    minElevation = minEl, maxElevation = maxEl, azimuthSpacing = azimuthSpacing, elevationSpacing = elevationSpacing )
            else:
                print "Color data cannot be derived from the sent data type: ", type( points )
                print "Panorama was not created"
                return None
            
        elif ( _property == 'segmentation' ):  # Creating a panorama based on the segmentation of point set
            
            if ( isinstance( points, SegmentationProperty ) ):  # Checking if the input 'points' object is an instance of SegmenationProperty
                return PanoramaProperty( pointSet, elevationIndexes, azimuthIndexes, points.RGB, dataType = _property, minAzimuth = minAz, maxAzimuth = maxAz,
                                    minElevation = minEl, maxElevation = maxEl, azimuthSpacing = azimuthSpacing, elevationSpacing = elevationSpacing )
            else:
                print "Segmentation data cannot be derived from the sent data type: ", type( points )
                print "Panorama was not created"
                return None
            
        elif ( _property == 'normals' ):  # Creating a panorama based on the normals of point set
            
            if ( isinstance( points, NormalsProperty ) ):  # Checking if the input 'points' object is an instance of NormalsProperty
                return PanoramaProperty( pointSet, elevationIndexes, azimuthIndexes, points.Normals, dataType = _property, minAzimuth = minAz, maxAzimuth = maxAz,
                                    minElevation = minEl, maxElevation = maxEl, azimuthSpacing = azimuthSpacing, elevationSpacing = elevationSpacing )
            else:
                print "Normals data cannot be derived from the sent data type: ", type( points )
                print "Panorama was not created"
                return None
                

if __name__ == '__main__':
    from IOFactory import IOFactory
    from Visualization import Visualization
    import matplotlib.pyplot as plt
    
    pointSetList = []
    fileName = r'D:\Documents\Pointsets\bonim_5_big.pts'
    
    print 'Reading from file'
    IOFactory.ReadPts( fileName, pointSetList )
    
    print 'Creating Panorama'
    pano = PanoramaFactory.CreatePanorama( pointSetList[0] )
    
    print 'Showing Panorama'
    Visualization.ShowPanorama( pano, 'gray' )
    
#     pano = PanoramaFactory.CreatePanorama( pointSetList[0], 'intensity' )
#     Visualization.ShowPanorama( pano, 'gray' )
    
    plt.show()
