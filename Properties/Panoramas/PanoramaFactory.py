from numpy import min, max, int
from warnings import warn

from DataClasses.PointSet import PointSet
from Properties.BaseProperty import BaseProperty
from Properties.Panoramas.PanoramaProperty import PanoramaProperty
from Properties.Transformations.SphericalCoordinatesFactory import SphericalCoordinatesFactory
from Properties.Transformations.SphericalCoordinatesProperty import SphericalCoordinatesProperty


class PanoramaFactory:
    """
    Creates a panoramic view from point set based on a certain property (e.g. range, intesity, etc.)
    The panoramic view is stored as a PanoramaProperty object
    """

    @classmethod
    def CreatePanorama_byPoints(cls, points, azimuthSpacing=0.057, elevationSpacing=0.057,
                                intensity=False, **kwargs):
        """
        Create a PanoramaProperty object from a point set based on according to range or intensity
        
        :param points: The point set to create the panorama from can be either a PointSet, PointSubSet or
           BaseProperty-derived objects
        :param azimuthSpacing: The spacing between two points of the point set in the azimuth direction (scan property)
        :param elevationSpacing: The spacing between two points of the point set in the elevation direction
        :param intensity: if the pixel's value should be the intensity value.

        :type points: PointSet
        :type azimuthSpacing: float
        :type elevationSpacing: float
        :type intensity: bool

         **Optionals**

        :param void_as_mean: flag to determine the void value as the mean value of the ranges

        :type void_as_mean: bool

        :return: panorama_property
        :rtype: PanoramaProperty

        """

        void_as_mean = kwargs.get('void_as_mean', False)

        try:
            # Calculating the spherical coordinates of the point set
            sphCoords = SphericalCoordinatesFactory.CartesianToSphericalCoordinates(points)
        except:
            warn('Did not convert to spherical coordinates')
            return 1

        (minAz, maxAz), (minEl, maxEl), (azimuthIndexes, elevationIndexes) = \
            cls.__computePanoramaIndices(sphCoords, azimuthSpacing=azimuthSpacing,
                                         elevationSpacing=elevationSpacing)

        # Create the panorama
        if not intensity:
            # range as pixel value
            panorama = PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, sphCoords.Ranges,
                                        minAzimuth=minAz, maxAzimuth=maxAz,
                                        minElevation=minEl, maxElevation=maxEl, azimuthSpacing=azimuthSpacing,
                                        elevationSpacing=elevationSpacing, **kwargs)
        else:
            #  intensity as pixel value
            panorama = PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, points.Intensity,
                                        minAzimuth=minAz, maxAzimuth=maxAz,
                                        minElevation=minEl, maxElevation=maxEl, azimuthSpacing=azimuthSpacing,
                                        elevationSpacing=elevationSpacing, **kwargs)
        if void_as_mean:
            void = cls.__compute_void_as_mean(sphCoords.Ranges)
            panorama.load(voidData=void)

        return panorama

    @classmethod
    def CreatePanorama_byProperty(cls, pointSet_property, azimuthSpacing=0.057, elevationSpacing=0.057,
                                  intensity=False, property_array=None, **kwargs):
        """
        Creates panorama with the property as the values of the pixels.

        :param pointSet_property: any PointSet property according to which the pixel's value should be
        :param azimuthSpacing: The spacing between two points of the point set in the azimuth direction (scan property)
        :param elevationSpacing: The spacing between two points of the point set in the elevation direction
        :param property_array: if the values of the property to present cannot be retrieved by a simple ``getValues()``

        .. code-block:: python

            PanoramaFactory.CreatePanorama_byProperty(curvatureProperty, azimuthSpacing=0.057, elevationSpacing=0.057,
                                  intensity=False, property_array=curvatureProperty.k1)

        :param intensity: if the pixel's value should be the intensity value.
        :param voidData: the number to set where there is no data
        :param void_as_mean: flag to determine the void value as the mean value of the ranges

        :type pointSet_property: BaseProperty
        :type azimuthSpacing: float
        :type elevationSpacing: float
        :type inensity: bool
        :type property_array: numpy.array
        :type void_as_mean: bool
        :type voidData: float

        :return: panorama_property
        :rtype: PanoramaProperty

        """
        import numpy as np

        void_as_mean = kwargs.get('void_as_mean', False)


        if isinstance(pointSet_property, SphericalCoordinatesProperty):
            sphCoords = pointSet_property

        else:
            sphCoords = SphericalCoordinatesFactory.CartesianToSphericalCoordinates(pointSet_property.Points)

        (minAz, maxAz), (minEl, maxEl), (azimuthIndexes, elevationIndexes) = \
            cls.__computePanoramaIndices(sphCoords, azimuthSpacing=azimuthSpacing,
                                         elevationSpacing=elevationSpacing)

        if np.all(property_array is not None):
            panorama = PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, property_array,
                                        minAzimuth=minAz, maxAzimuth=maxAz,
                                        minElevation=minEl, maxElevation=maxEl, azimuthSpacing=azimuthSpacing,
                                        elevationSpacing=elevationSpacing, **kwargs)
        elif not intensity:
            panorama = PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, pointSet_property.getValues(),
                                        minAzimuth=minAz, maxAzimuth=maxAz,
                                        minElevation=minEl, maxElevation=maxEl, azimuthSpacing=azimuthSpacing,
                                        elevationSpacing=elevationSpacing, **kwargs)


        else:
            panorama = PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, pointSet_property.Points.Intensity,
                                        minAzimuth=minAz, maxAzimuth=maxAz,
                                        minElevation=minEl, maxElevation=maxEl, azimuthSpacing=azimuthSpacing,
                                        elevationSpacing=elevationSpacing, **kwargs)

        if void_as_mean:
            void = cls.__compute_void_as_mean(sphCoords.Ranges)
            panorama.load(voidData=void)

        return panorama

    @classmethod
    def __compute_void_as_mean(cls, ranges):
        """
        Sets the void value as the mean value of the ranges

        """
        import numpy as np
        void_value = np.mean(ranges)
        return void_value

    @staticmethod
    def __computePanoramaIndices(sphCoords, azimuthSpacing, elevationSpacing):
        """
        Find the boundaries and the indices of the panorama

        :return: (minAz, maxAz), (minEl, maxEl), (azimuthIndexes, elevationIndexes)

        :rtype: tuple

        """

        # Finding the boundaries of the panorama
        minAz = min(sphCoords.Azimuths)
        maxAz = max(sphCoords.Azimuths)
        minEl = min(sphCoords.ElevationAngles)
        maxEl = max(sphCoords.ElevationAngles)

        # Calculating the location of each point in the panorama
        azimuthIndexes = ((sphCoords.Azimuths - minAz) / azimuthSpacing).astype(int)
        elevationIndexes = ((maxEl - sphCoords.ElevationAngles) / elevationSpacing).astype(int)

        return (minAz, maxAz), (minEl, maxEl), (azimuthIndexes, elevationIndexes)
