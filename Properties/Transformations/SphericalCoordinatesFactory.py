import numexpr as ne
import numpy as np
from numpy import pi

from DataClasses.PointSet import PointSet
from SphericalCoordinatesProperty import SphericalCoordinatesProperty


class SphericalCoordinatesFactory:
    """
    SphericalCoordinatesFactory
    """    
    
    @staticmethod
    def CartesianToSphericalCoordinates(points, ceval=ne.evaluate):
        """
        CartesianToSphericalCoordinates

        :param points: a pointset containing the points to transform to spherical
        :param ceval: backend to use:
                 - eval: pure Numpy
                 - numexpr.evaluate (default): Numexpr (faster for large arrays)

        :return: spherical coordinates property in degrees
        :rtype: SphericalCoordinatesProperty
        """

        from numpy import arctan2, sqrt

        x = points.X
        y = points.Y
        z = points.Z

        # temp vars so the IDE code optimization won't delete the import
        try:
            temp1 = arctan2(x[0], y[0])
            temp2 = sqrt(np.abs(z[0]))
        except RuntimeError:
            print('Points are empty, it is not possible to convert')

        azimuth = ceval('arctan2(y,x)')
        elevation = ceval('arctan2(z, sqrt(x**2+y**2))')
        range = ceval('sqrt(x**2+y**2+z**2)')

        elevation = np.rad2deg(elevation)
        azimuth = np.rad2deg(azimuth)
        azimuth[azimuth < 0] = 360. + azimuth[azimuth < 0]

        return SphericalCoordinatesProperty(points, azimuth, elevation, range)

    @staticmethod
    def cart2sph_elementwise(x, y, z):
        """
        Cartesian to spherical transformation for RDD use.

        :param x:
        :param y:
        :param z:

        :return: spherical coordinates
        
        """

        azimuth = np.arctan2(y, x)
        xy2 = x ** 2 + y ** 2
        elevation = np.arctan2(z, np.sqrt(xy2))
        range = np.sqrt(xy2 + z ** 2)

        elevation *= 180. / np.pi
        azimuth *= 180. / np.pi
        if azimuth < 0:
            azimuth += 360.

        return float(elevation), float(azimuth), float(range)

    def cart2sph_RDD(self, points, sc):
        """
        Cartesian to spherical transformation via RDD.

        :param points: a point set to be transformed
        :param sc: a spark context object (driver) that will run the job

        :type points: PointSet
        :type SparkContext

        :return: RDD holding the spherical coordinates

        :rtype: pySpark RDD

        """

        return points.ToRDD().map(lambda y: self.cart2sph_elementwise(float(y[0]), float(y[1]), float(y[2])))
