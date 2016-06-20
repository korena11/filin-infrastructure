from BaseProperty import BaseProperty
from numpy import zeros, arange, array
from tvtk.api import tvtk


class SphericalCoordinatesProperty(BaseProperty):
    '''
    Class for containing the spherical coordinates of a certain PointSet Object
    Angular coordinates are in decimal degrees
    '''
    
    def __init__(self, points, azimuths, elevationAngles, distances):
        
        self._BaseProperty__points = points
        self.__azimuthElevationRange = zeros((points.Size, 3))
        self.__azimuthElevationRange[:, 0] = azimuths
        self.__azimuthElevationRange[:, 1] = elevationAngles
        self.__azimuthElevationRange[:, 2] = distances 
    
    @property
    def XYZ(self):
        
        return self._BaseProperty__points
        
    @property
    def Size(self):
        
        return self._BaseProperty__points.Size
    
    @property
    def Azimuths(self):
        
        return self.__azimuthElevationRange[:, 0]
    
    @property
    def ElevationAngles(self):
        
        return self.__azimuthElevationRange[:, 1]
    
    @property
    def Ranges(self):
        
        return self.__azimuthElevationRange[:, 2]
    
    
    def ToNumpy(self):
        
        return self.__azimuthElevationRange
    
   
    def ToPolyData(self):
        """
        Create and return PolyData object
        
        :Return:
            - tvtk.PolyData of the current PointSet
        """
  
        _polyData = tvtk.PolyData(points=array(self.__azimuthElevationRange, 'f'))
        verts = arange(0, self.__azimuthElevationRange.shape[0], 1)
        verts.shape = (self.__azimuthElevationRange.shape[0], 1)
        _polyData.verts = verts
        
        return _polyData
