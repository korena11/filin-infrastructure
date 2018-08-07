import numpy as np
from numpy import ones

from BaseProperty import BaseProperty
from PointSubSet import PointSubSet
from SphericalCoordinatesProperty import SphericalCoordinatesProperty


class PanoramaProperty(BaseProperty):
    """
    A panoramic representation of the point set
    """

    __rowIndexes = None  # An array of indexes corresponding to the row number to which each point belongs to
    __columnIndexes = None  # An array of indexes corresponding to the column number to which each point belongs to
    __panoramaData = None  # A m-by-n-by-p array in which the panorama is stored
    __panoramaIndex = None  # A m-by-n-by-p array in which the indices of the points are stored
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
        :param panoramaData: data to be represented as a panorama (e.g. range, intensity, etc.). Default: range
        :param rowIndexes: row indices of the points in the point set based on the elevation angles
        :param columnIndexes: column indices of the points in the point set based on the azimuth angles


        :type sphericalCoordinates: SphericalCoordinatesProperty
        :type rowIndexes: int
        :type columnIndexes: int
        :type panoramaData: np.array

        """
        super(PanoramaProperty, self).__init__(sphericalCoordinates.Points)

        self.setValues(**kwargs)

        self.__columnIndexes = columnIndexes
        self.__rowIndexes = rowIndexes

        numRows = int((self.max_elevation - self.min_elevation) / self.elevation_spacing) + 1
        numColumns = int((self.max_azimuth - self.min_azimuth) / self.azimuth_spacing) + 1

        if len(panoramaData.shape) == 1:
            self.__panoramaData = self.void_data * ones((numRows, numColumns))
            self.__panoramaData[rowIndexes, columnIndexes] = panoramaData
        else:
            self.__panoramaData = self.void_data * ones((numRows, numColumns, panoramaData.shape[1]))
            self.__panoramaData[rowIndexes, columnIndexes, :] = panoramaData[:, :]

    def indexes_to_panorama(self):
        """
        Arrange the points' indices into the panorama structure

        """
        # set so that unfilled cells will be NaN
        panoramaIndex = np.empty(self.getValues.shape, dtype = np.int)
        panoramaIndex[:] = np.inf

        pts_index = np.arange(self.Points.Size)
        panoramaIndex[self.row_indexes, self.column_indexes] = pts_index

        self.__panoramaIndex = panoramaIndex
        return panoramaIndex



    @property
    def PanoramaImage(self):
        """
        Returns the panorama image
        """
        return self.__panoramaData

    @property
    def azimuth_spacing(self):
        """
        Azimuth angle resolution size

        """
        return self.__azimuthSpacing

    @property
    def elevation_spacing(self):
        """
        Elevation angle resolution size

        """
        return self.__elevationSpacing

    @property
    def column_indexes(self, *args):
        """
        The column index of each point

        """

        return self.__columnIndexes

    @property
    def row_indexes(self):
        """
        The row index of each point

        """
        return self.__rowIndexes

    @property
    def max_azimuth(self):
        """
        Maximal azimuth value in data

        """

        return self.__maxAzimuth

    @property
    def min_azimuth(self):
        """
        Minimal azimuth value in data

        """

        return self.__minAzimuth

    @property
    def max_elevation(self):
        """
        Maximal elevation value in data

        """

        return self.__maxElevation

    @property
    def min_elevation(self):
        """
        Minimal elevation value in data

        """

        return self.__minElevation

    @property
    def getValues(self):
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

    @property
    def void_data(self):
        """
        Value set for void areas

        """
        return self.__voidData

    def extract_area(self, left_top_corner, right_bottom_corner):
        """
        Create a new panorama according to a bounding box

        :param left_top_corner: (row, column) of the left top corner of the extracted area
        :param right_bottom_corner: (row, column) of the right bottom corner of the extracted area

        :type left_top_corner: tuple
        :type right_bottom_corner: tuple

        :return: a pointsubset according to the bounding box
        :rtype: PointSubSet
        """

        row_ind = np.nonzero((left_top_corner[0] < self.row_indexes) * (self.row_indexes < right_bottom_corner[0]))
        col_ind = np.nonzero((left_top_corner[1] < self.column_indexes) *
                             (self.column_indexes < right_bottom_corner[1]))

        ind = np.intersect1d(row_ind, col_ind)
        return PointSubSet(self.Points, ind)

