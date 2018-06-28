from numpy import ones

from BaseProperty import BaseProperty
from SphericalCoordinatesProperty import SphericalCoordinatesProperty


class PanoramaProperty(BaseProperty):
    """
    A panoramic representation of the point set
    """

    __rowIndexes = None  # An array of indexes corresponding to the row number to which each point belongs to
    __columnIndexes = None  # An array of indexes corresponding to the column number to which each point belongs to
    __panoramaData = None  # A m-by-n-by-p array in which the panorama is stored
    __voidData = 250  # A number indicating missing data in the panorama
    __minAzimuth = 0  # The minimal azimuth value
    __maxAzimuth = 360  # The maximal azimuth value
    __minElevation = -45  # The minimal elevation angle value
    __maxElevation = 90  # The maximal elevation angle value
    __azimuthSpacing = 0.057  # The spacing between points in the azimuth direction
    __elevationSpacing = 0.057  # The spacing between points in the elevation angle direction  

    def __init__(self, sphericalCoordinates, rowIndexes = None, columnIndexes = None, panoramaData = None, **kwargs):
        """
        Constuctor - Creates a panoramic view of the data sent

        :param sphericalCoordinates: SphericalCoordinates property
        :param panoramaData: The data to be represented as a panorama (e.g. range, intesity, etc.). Default: range
        :param rowIndexes: The row indices of the points in the point set based on the elevation angles
        :param columnIndexes: The column indices of the points in the point set based on the azimuth angles


        :type sphericalCoordinates: SphericalCoordinatesProperty
        :type rowIndexes: int
        :type columnIndexes: int
        :type panoramaData: np.array

        """
        super(PanoramaProperty, self).__init__(sphericalCoordinates.Points)

        self.setValues(**kwargs)

        numRows = int((self.__maxElevation - self.__minElevation) / self.__elevationSpacing) + 1
        numColumns = int((self.__maxAzimuth - self.__minAzimuth) / self.__azimuthSpacing) + 1

        if (len(panoramaData.shape) == 1):
            self.__panoramaData = self.__voidData * ones((numRows, numColumns))
            self.__panoramaData[rowIndexes, columnIndexes] = panoramaData
        else:
            self.__panoramaData = self.__voidData * ones((numRows, numColumns, panoramaData.shape[1]))
            self.__panoramaData[rowIndexes, columnIndexes, :] = panoramaData[:, :]

    @property
    def PanoramaImage(self):
        """
        Returns the panorama image
        """
        return self.__panoramaData

    def setValues(self, **kwargs):
        """
        Sets values into the panoramaProperty object

        :param panoramaData: The data to be represented as a panorama (e.g. range, intensity, etc.). Default: range
        :param rowIndexes: The row indices of the points in the point set based on the elevation angles
        :param columnIndexs: The column indices of the points in the point set based on the azimuth angles
        :param minAzimuth: The minimal azimuth value
        :param maxAzimuth: The maximal azimuth value
        :param minElevation: The minimal elevation value
        :param maxElevation: The maximal elevation value
        :param azimuthSpacing: The measurements' angular resolution in the azimuth direction.
        :param elevationSpacing:  The measurements' angular resolution in the elevation angle direction

        :type rowIndexes: int
        :type columnIndexes: int
        :type panoramaData: np.array
        :type dataType: str
        :type minAzimuth: float
        :type maxAzimuth: float
        :type minElevation: float
        :type maxElevation: float
        :type azimuthSpacing: float
        :type elevationSpacing: float

        .. note:: For the Scanstation C10 the measurements' angular resolution for both elevation and azimuth directions:

            * Low: 0.11 deg
            * Medium: 0.057 deg
            * High: 0.028 deg
            * Highest: *TO ADD*


        """

        self.__maxAzimuth = kwargs.get('maxAzimuth', self.__maxAzimuth)
        self.__minAzimuth = kwargs.get('minAzimuth', self.__minAzimuth)
        self.__minElevation = kwargs.get('minElevation', self.__minElevation)
        self.__maxElevation = kwargs.get('maxElevation', self.__maxElevation)

        self.__azimuthSpacing = kwargs.get('azimuthSpacing', self.__azimuthSpacing)
        self.__elevationSpacing = kwargs.get('elevationSpacing', self.__elevationSpacing)
        self.__voidData = kwargs.get('voidData', self.__voidData)

        self.__columnIndexes = kwargs.get('columnIndexes', self.__columnIndexes)
        self.__rowIndexes = kwargs.get('rowIndexes', self.__rowIndexes)
        self.__panoramaData = kwargs.get('panoramaData', self.__panoramaData)
