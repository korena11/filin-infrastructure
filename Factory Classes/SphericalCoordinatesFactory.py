from numpy import sqrt, pi, arctan2, cos, sin, zeros
from SphericalCoordinatesProperty import SphericalCoordinatesProperty
from PointSet import PointSet

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
    
    @staticmethod
    def SphericalToCartesianCoordinates(points):
        """
        SphericalToCartesainCoordinates
        :Args:
            - points: spherical coordinates (az,el,r)
        """
        x = points[:, 2] * cos(points[:, 1] * pi / 180) * cos(points[:, 0] * pi / 180) 
        y = points[:, 2] * cos(points[:, 1] * pi / 180) * sin(points[:, 0] * pi / 180)
        z = points[:, 2] * sin(points[:, 1] * pi / 180)
        
        xyz = zeros((len(x), 3))
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = z
        
        return PointSet(xyz)
        
        
