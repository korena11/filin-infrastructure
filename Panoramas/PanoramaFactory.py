from warnings import warn

from numpy import min, max, int

from BaseProperty import BaseProperty
from Panoramas.PanoramaProperty import PanoramaProperty
from PointSet import PointSet
from SphericalCoordinatesFactory import SphericalCoordinatesFactory
from SphericalCoordinatesProperty import SphericalCoordinatesProperty


class PanoramaFactory:
    """
    Creates a panoramic view from point set based on a certain property (e.g. range, intesity, etc.)
    The panoramic view is stored as a PanoramaProperty object
    """

    @classmethod
    def CreatePanorama_byPoints(cls, points, **kwargs):
        """
        Creating a PanoramaProperty object from a point set based on certain property
        
        :param points: The point set to create the panorama from can be either a PointSet, PointSubSet or
        BaseProperty-derived objects

        :param azimuthSpacing: The spacing between two points of the point set in the azimuth direction (scan property)
        :param elevationSpacing: The spacing between two points of the point set in the elevation direction
        :param intensity: if the pixel's value should be the intensity value.

        :type points: PointSet
        :type azimuthSpacing: float
        :type elevationSpacing: float
        :type intensity: bool

        :return: panorama_property
        :rtype: PanoramaProperty

        """
        azimuthSpacing = kwargs.get('azimuthSpacing', 0.057)
        elevationSpacing = kwargs.get('elevationSpacing', 0.057)
        intensity = kwargs.get('intensity', False)

        try:
            # Calculating the spherical coordinates of the point set
            sphCoords = SphericalCoordinatesFactory.CartesianToSphericalCoordinates(points)
        except:
            warn('Expected PointSet, got property instead')
            return 1

        (minAz, maxAz), (minEl, maxEl), (azimuthIndexes, elevationIndexes) = \
            cls.__computePanoramaIndices(sphCoords, azimuthSpacing = azimuthSpacing,
                                         elevationSpacing = elevationSpacing)

        # Create the panorama
        if not intensity:
            # range as pixel value
            return PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, sphCoords.Ranges,
                                    minAzimuth = minAz, maxAzimuth = maxAz,
                                    minElevation = minEl, maxElevation = maxEl, **kwargs)

        else:
            #  intensity as pixel value
            return PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, points.Intensity,
                                    minAzimuth = minAz, maxAzimuth = maxAz,
                                    minElevation = minEl, maxElevation = maxEl, **kwargs)

    @classmethod
    def CreatePanorama_byProperty(cls, pointSet_property, **kwargs):
        """
        Creates panorama with the property as the values of the pixels.

        :param pointSet_property: any PointSet property according to which the pixel's value should be
        :param azimuthSpacing: The spacing between two points of the point set in the azimuth direction (scan property)
        :param elevationSpacing: The spacing between two points of the point set in the elevation direction
        :param intensity: if the pixel's value should be the intensity value.

        :type pointSet_property: BaseProperty
        :type azimuthSpacing: float
        :type elevationSpacing: float
        :type inensity: bool

        :return: panorama_property
        :rtype: PanoramaProperty

        """
        azimuthSpacing = kwargs.get('azimuthSpacing', 0.057)
        elevationSpacing = kwargs.get('elevationSpacing', 0.057)
        intensity = kwargs.get('intensity', False)

        if isinstance(pointSet_property, SphericalCoordinatesProperty):
            sphCoords = pointSet_property

        else:
            sphCoords = SphericalCoordinatesFactory.CartesianToSphericalCoordinates(pointSet_property.Points)

        (minAz, maxAz), (minEl, maxEl), (azimuthIndexes, elevationIndexes) = \
            cls.__computePanoramaIndices(sphCoords, azimuthSpacing = azimuthSpacing,
                                         elevationSpacing = elevationSpacing)

        if not intensity:
            return PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, pointSet_property.getValues(),
                                    minAzimuth = minAz, maxAzimuth = maxAz,
                                    minElevation = minEl, maxElevation = maxEl, azimuthSpacing = azimuthSpacing,
                                    elevationSpacing = elevationSpacing)

        else:
            return PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, pointSet_property.Points.Intensity,
                                    minAzimuth = minAz, maxAzimuth = maxAz,
                                    minElevation = minEl, maxElevation = maxEl, azimuthSpacing = azimuthSpacing,
                                    elevationSpacing = elevationSpacing)

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
