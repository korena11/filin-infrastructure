from numpy import min, max, int
from warnings import warn
import numpy as np

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
    def CreatePanorama(cls, points, azimuthSpacing=0.057, elevationSpacing=0.057,  property_array=None, **kwargs):
        """
        Create a PanoramaProperty object from a point set.
        
        :param points: The point set to create the panorama from can be either a PointSet, PointSubSet or
           BaseProperty-derived objects
        :param azimuthSpacing: The spacing between two points of the point set in the azimuth direction (scan property)
        :param elevationSpacing: The spacing between two points of the point set in the elevation direction
        :param property_array: if the values of the property to present cannot be retrieved by a simple ``getValues()``
        :param voidData: the value for NoData. Default: 250 (meters)

        :type points: PointSet or BaseProperty
        :type azimuthSpacing: float
        :type elevationSpacing: float
        :type property_array: numpy.array
        :type voidData: float

        :return: panorama_property
        :rtype: PanoramaProperty

        """
        # Calculate spherical coordinates of the point set
        if isinstance(points, SphericalCoordinatesProperty):
            sphCoords = points
        else:
            try:
                if isinstance(points, BaseProperty):
                    sphCoords = SphericalCoordinatesFactory.CartesianToSphericalCoordinates(points.Points)
                else:
                    sphCoords = SphericalCoordinatesFactory.CartesianToSphericalCoordinates(points)

            except AttributeError:
                warn('Did not convert to spherical coordinates')
                return 1

        (minAz, maxAz), (minEl, maxEl), (azimuthIndexes, elevationIndexes) = \
            cls.__computePanoramaIndices(sphCoords, azimuthSpacing=azimuthSpacing,
                                         elevationSpacing=elevationSpacing)

        panorama = None

        if np.all(property_array is not None):
            panorama = PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes,
                                        panoramaData=property_array, intensityData=points.Points.Intensity,
                                        minAzimuth=minAz, maxAzimuth=maxAz,
                                        minElevation=minEl, maxElevation=maxEl, azimuthSpacing=azimuthSpacing,
                                        elevationSpacing=elevationSpacing, **kwargs)

        elif isinstance(points, BaseProperty):
            panorama = PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes,
                                        panoramaData=points.getValues(),intensityData= points.Points.Intensity,
                                        minAzimuth=minAz, maxAzimuth=maxAz,
                                        minElevation=minEl, maxElevation=maxEl, azimuthSpacing=azimuthSpacing,
                                        elevationSpacing=elevationSpacing, **kwargs)
        elif isinstance(points, PointSet):
            panorama = PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, None, intensityData=points.Intensity,
                                        minAzimuth=minAz, maxAzimuth=maxAz,
                                        minElevation=minEl, maxElevation=maxEl, azimuthSpacing=azimuthSpacing,
                                        elevationSpacing=elevationSpacing, **kwargs)
        # TODO: add for other types
        else:
            warn('unexpected type')


        return panorama

    # @classmethod
    # def CreatePanorama_byProperty(cls, pointSet_property, azimuthSpacing=0.057, elevationSpacing=0.057,
    #                               intensity=False, property_array=None, **kwargs):
    #     """
    #     Creates panorama with the property as the values of the pixels.
    #
    #     :param pointSet_property: any PointSet property according to which the pixel's value should be
    #     :param azimuthSpacing: The spacing between two points of the point set in the azimuth direction (scan property)
    #     :param elevationSpacing: The spacing between two points of the point set in the elevation direction
    #     :param property_array: if the values of the property to present cannot be retrieved by a simple ``getValues()``
    #     :param intensity: if the pixel's value should be the intensity value.
    #     :param voidData: the number to set where there is no data
    #     :param void_as_mean: flag to determine the void value as the mean value of the ranges
    #
    #     :type pointSet_property: BaseProperty
    #     :type azimuthSpacing: float
    #     :type elevationSpacing: float
    #     :type inensity: bool
    #     :type property_array: numpy.array
    #     :type void_as_mean: bool
    #     :type voidData: float
    #
    #     .. code-block:: python
    #
    #         PanoramaFactory.CreatePanorama_byProperty(curvatureProperty, azimuthSpacing=0.057, elevationSpacing=0.057,
    #                               intensity=False, property_array=curvatureProperty.k1)
    #
    #
    #     :return: panorama_property
    #     :rtype: PanoramaProperty
    #
    #     """
    #     import numpy as np
    #
    #     void_as_mean = kwargs.get('void_as_mean', False)
    #
    #
    #     if isinstance(pointSet_property, SphericalCoordinatesProperty):
    #         sphCoords = pointSet_property
    #
    #     else:
    #         sphCoords = SphericalCoordinatesFactory.CartesianToSphericalCoordinates(pointSet_property.Points)
    #
    #     (minAz, maxAz), (minEl, maxEl), (azimuthIndexes, elevationIndexes) = \
    #         cls.__computePanoramaIndices(sphCoords, azimuthSpacing=azimuthSpacing,
    #                                      elevationSpacing=elevationSpacing)
    #
    #     if np.all(property_array is not None):
    #         panorama = PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, property_array,
    #                                     minAzimuth=minAz, maxAzimuth=maxAz,
    #                                     minElevation=minEl, maxElevation=maxEl, azimuthSpacing=azimuthSpacing,
    #                                     elevationSpacing=elevationSpacing, **kwargs)
    #     elif not intensity:
    #         panorama = PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, pointSet_property.getValues(),
    #                                     minAzimuth=minAz, maxAzimuth=maxAz,
    #                                     minElevation=minEl, maxElevation=maxEl, azimuthSpacing=azimuthSpacing,
    #                                     elevationSpacing=elevationSpacing, **kwargs)
    #
    #
    #     else:
    #         panorama = PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, pointSet_property.Points.Intensity,
    #                                     minAzimuth=minAz, maxAzimuth=maxAz,
    #                                     minElevation=minEl, maxElevation=maxEl, azimuthSpacing=azimuthSpacing,
    #                                     elevationSpacing=elevationSpacing, **kwargs)
    #
    #     if void_as_mean:
    #         void = cls.__compute_void_as_mean(sphCoords.ranges)
    #         panorama.load(voidData=void)
    #
    #     return panorama

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
        import numpy as np

        # Finding the boundaries of the panorama
        minAz = min(sphCoords.azimuths)
        maxAz = max(sphCoords.azimuths)
        minEl = min(sphCoords.elevations)
        maxEl = max(sphCoords.elevations)

        # Calculating the location of each point in the panorama
        azimuthIndexes = ((sphCoords.azimuths - minAz) / azimuthSpacing).astype(int)
        elevationIndexes = ((maxEl - sphCoords.elevations) / elevationSpacing).astype(int)

        return (minAz, maxAz), (minEl, maxEl), (azimuthIndexes, elevationIndexes)
