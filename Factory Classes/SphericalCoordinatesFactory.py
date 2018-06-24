from numpy import sqrt, pi, arctan2

from SphericalCoordinatesProperty import SphericalCoordinatesProperty


class SphericalCoordinatesFactory:
    """
    SphericalCoordinatesFactory
    """    
    
    @staticmethod
    def CartesianToSphericalCoordinates(points):
        """
        CartesianToSphericalCoordinates
        """
        
        horizontalSquaredDistance = points.X ** 2 + points.Y ** 2
        
        dis = sqrt(horizontalSquaredDistance + points.Z ** 2)
        
        el = arctan2(points.Z, sqrt(horizontalSquaredDistance)) * 180 / pi
         
        az = arctan2(points.Y, points.X) * 180 / pi
        az[az < 0 ] = 360 + az[az < 0]
        
        return SphericalCoordinatesProperty(points, az, el, dis)
