from warnings import warn

from numpy import min, max, int

from BaseProperty import BaseProperty
from ColorProperty import ColorProperty
from NormalsProperty import NormalsProperty
from Panoramas.PanoramaProperty import PanoramaProperty
from PointSet import PointSet
from SegmentationProperty import SegmentationProperty
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
        self.azimuthSpacing = kwargs.get('azimuthSpacing', 0.057)
        elevationSpacing = kwargs.get('elevationSpacing', 0.057)
        intensity = kwargs.get('intensity', False)

        try:
            # Calculating the spherical coordinates of the point set
            sphCoords = SphericalCoordinatesFactory.CartesianToSphericalCoordinates(points)
        except:
            warn('Expected PointSet, got property instead')
            return 1




        # Creating the panorams based on the requested property
        if not intensity:
            return PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, sphCoords.Ranges,
                                    panoramaData = property, minAzimuth = minAz, maxAzimuth = maxAz,
                                    minElevation = minEl, maxElevation = maxEl, azimuthSpacing = azimuthSpacing,
                                    elevationSpacing = elevationSpacing)

        else:  # Creating an inensity panorama
            return PanoramaProperty(sphCoords, elevationIndexes, azimuthIndexes, points.Intensity,
                                    panoramaData = property, minAzimuth = minAz, maxAzimuth = maxAz,
                                    minElevation = minEl, maxElevation = maxEl, azimuthSpacing = azimuthSpacing,
                                    elevationSpacing = elevationSpacing)

    @classmethod
    def CreatePanorama_byProperty(cls, property, **kwargs):
        """
        Creates panorama with the property as the values of the pixels.

        :param property: any PointSet property according to which the pixel's value should be
        :param azimuthSpacing: The spacing between two points of the point set in the azimuth direction (scan property)
        :param elevationSpacing: The spacing between two points of the point set in the elevation direction
        :param intensity: if the pixel's value should be the intensity value.

        :type property: BaseProperty
        :type azimuthSpacing: float
        :type elevationSpacing: float
        :type inensity: bool

        :return: panorama_property
        :rtype: PanoramaProperty

        """
        azimuthSpacing = kwargs.get('azimuthSpacing', 0.057)
        elevationSpacing = kwargs.get('elevationSpacing', 0.057)
        intensity = kwargs.get('intensity', False)

        if isinstance(property, SphericalCoordinatesProperty):
            sphCoords = property

        else:
            sphCoords = SphericalCoordinatesFactory.CartesianToSphericalCoordinates(property.Points)

        if not intensity:
            return PanoramaProperty(property.Points, )


        elif (property == 'color'):  # Creating a panorama based on the colors of point set
            # Checking if the input 'points' object is an instance of either ColorProperty or SegmenationProperty which have color data
            if (isinstance(points, ColorProperty) or isinstance(points, SegmentationProperty)):
                return PanoramaProperty(points, elevationIndexes, azimuthIndexes, points.RGB,
                                        panoramaData = property, minAzimuth = minAz, maxAzimuth = maxAz,
                                        minElevation = minEl, maxElevation = maxEl, azimuthSpacing = azimuthSpacing,
                                        elevationSpacing = elevationSpacing)

            # Checking if the original PointSet has color data in it
            elif (pointSet.RGB != None):
                return PanoramaProperty(pointSet, elevationIndexes, azimuthIndexes, pointSet.RGB,
                                        panoramaData = _property, minAzimuth = minAz, maxAzimuth = maxAz,
                                        minElevation = minEl, maxElevation = maxEl, azimuthSpacing = azimuthSpacing,
                                        elevationSpacing = elevationSpacing)
            else:
                print("Color data cannot be derived from the sent data type: ", type(points))
                print("Panorama was not created")
                return None

        elif (_property == 'segmentation'):  # Creating a panorama based on the segmentation of point set

            if (isinstance(points,
                           SegmentationProperty)):  # Checking if the input 'points' object is an instance of SegmenationProperty
                return PanoramaProperty(pointSet, elevationIndexes, azimuthIndexes, points.RGB,
                                        panoramaData = _property, minAzimuth = minAz, maxAzimuth = maxAz,
                                        minElevation = minEl, maxElevation = maxEl, azimuthSpacing = azimuthSpacing,
                                        elevationSpacing = elevationSpacing)
            else:
                print("Segmentation data cannot be derived from the sent data type: ", type(points))
                print("Panorama was not created")
                return None

        elif (_property == 'normals'):  # Creating a panorama based on the normals of point set

            if (isinstance(points,
                           NormalsProperty)):  # Checking if the input 'points' object is an instance of NormalsProperty
                return PanoramaProperty(pointSet, elevationIndexes, azimuthIndexes, points.Normals,
                                        panoramaData = _property, minAzimuth = minAz, maxAzimuth = maxAz,
                                        minElevation = minEl, maxElevation = maxEl, azimuthSpacing = azimuthSpacing,
                                        elevationSpacing = elevationSpacing)
            else:
                print("Normals data cannot be derived from the sent data type: ", type(points))
                print("Panorama was not created")
                return None

    @staticmethod
    def __computePanoramaIndices(sphCoords, azimuthSpacing, elevationSpacing):
        """
        Find the boundaries and the indices of the panorama

        :return: tuple
        """

        # Finding the boundaries of the panorama
        minAz = min(sphCoords.Azimuths)
        maxAz = max(sphCoords.Azimuths)
        minEl = min(sphCoords.ElevationAngles)
        maxEl = max(sphCoords.ElevationAngles)

        # Calculating the location of each point in the panorama
        azimuthIndexes = ((sphCoords.Azimuths - minAz) / azimuthSpacing).astype(int)
        elevationIndexes = ((maxEl - sphCoords.ElevationAngles) / elevationSpacing).astype(int)

        return

if __name__ == '__main__':
    from IOFactory import IOFactory
    from Visualization import Visualization
    import matplotlib.pyplot as plt

    pointSetList = []
    fileName = r'D:\Documents\Pointsets\bonim_5_big.pts'

    print('Reading from file')
    IOFactory.ReadPts(fileName, pointSetList)

    print('Creating Panorama')
    pano = PanoramaFactory.CreatePanorama(pointSetList[0])

    print('Showing Panorama')
    Visualization.ShowPanorama(pano, 'gray')

    #     pano = PanoramaFactory.CreatePanorama( pointSetList[0], 'intensity' )
    #     Visualization.ShowPanorama( pano, 'gray' )

    plt.show()
