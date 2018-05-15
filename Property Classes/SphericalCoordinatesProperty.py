from numpy import zeros, arange, array
from tvtk.api import tvtk

from BaseProperty import BaseProperty


class SphericalCoordinatesProperty(BaseProperty):
    '''
    Spherical coordinates property of a certain PointSet Object
    Angular coordinates are in decimal degrees
    '''

    def __init__(self, points, *args):
        """

        :param points:
        :param azimuths:
        :param elevationAngles:
        :param distances:

        """

        super(SphericalCoordinatesProperty, self).__init__(points)
        self.setValues(*args)

    def setValues(self, *args, **kwargs):
        """
        Sets the values to the spherical coordinates property

        :param azimuths:
        :param elevationAngles:
        :param distances:

        :type azimuths: float
        :type elevationAngles: float
        :type distances: float

        """

        azimuths = args[0]
        elevationAngles = args[1]
        distances = args[2]
        self.__azimuthElevationRange = zeros((self.Points.Size, 3))
        self.__azimuthElevationRange[:, 0] = azimuths
        self.__azimuthElevationRange[:, 1] = elevationAngles
        self.__azimuthElevationRange[:, 2] = distances

    @property
    def XYZ(self):
        return self.Points.ToNumpy()

    @property
    def Size(self):
        return self.__dataset.Size
    
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
        
        :return: tvtk.PolyData of the current PointSet
            -
        """
  
        _polyData = tvtk.PolyData(points=array(self.__azimuthElevationRange, 'f'))
        verts = arange(0, self.__azimuthElevationRange.shape[0], 1)
        verts.shape = (self.__azimuthElevationRange.shape[0], 1)
        _polyData.verts = verts
        
        return _polyData
