import cv2
import numpy as np
from numpy import ones

from BaseProperty import BaseProperty
from NormalsFactory import NormalsFactory
from PointSet import PointSet
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

    def __init__(self, sphericalCoordinates, rowIndexes=None, columnIndexes=None, panoramaData=None, **kwargs):
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

    # def indexes_to_panorama(self):
    #     """
    #     Arrange the points' indices into the panorama structure
    #
    #     """
    #     # set so that unfilled cells will be NaN
    #     panoramaIndex = np.empty(self.getValues.shape, dtype = np.int)
    #     panoramaIndex[:] = np.inf
    #
    #     pts_index = np.arange(self.Points.Size)
    #     panoramaIndex[self.row_indexes, self.column_indexes] = pts_index
    #
    #     self.__panoramaIndex = panoramaIndex
    #     return panoramaIndex

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

        :TODO: Return panorama property of the subset and not the subset
        """

        row_ind = np.nonzero((left_top_corner[0] < self.row_indexes) * (self.row_indexes < right_bottom_corner[0]))
        col_ind = np.nonzero((left_top_corner[1] < self.column_indexes) *
                             (self.column_indexes < right_bottom_corner[1]))

        ind = np.intersect1d(row_ind, col_ind)
        return PointSubSet(self.Points, ind)

    def normals(self, phenomenaSize=0.2, ksize=(5, 5), gradientType='L1', **kwargs):
        r"""
        Computing gradients and normals in each direction, via adaptive smoothing according to :cite:`Arav2013`

        :param phenomenaSize: phenomena size for adaptive smoothing (default: 0.12 m)
        :param ksize: kernel size for gradient computation (default: (5,5))
        :param gradientType: 'L1' L1 norm of grad(I); 'L2' L2-norm of grad(I); 'LoG' Laplacian of gaussian (default: 'L1')

        **Optionals**

        :param rangeThresh: scanner's maximum range
        :param smoothing_function: if other smoothing than adaptive is required (e.g. 'guassianBlur'))

        :type ksize: tuple of int
        :type gradientType: str
        :type rangeThresh: float
        :type phenomenaSize: float

        :return:
            - normals: a holding  :math:`n\times m \times 3` ndarray of the normals in each direction (Nx, Ny, Nz)
            - gradient: gradient in each pixel
            - filtered: range image after adaptive smoothing
            - xyz: points' list after smoothing, from range image

        :rtype: tuple

        .. warning::

           Implemented for gaussian filtering and adaptive filters. Other adaptations might be
           needed when using different methods for smoothing.

        .. seealso:: method :py:meth:`NormalsFactory.NormalsFactory.normalsComputation_in_raster`
        """
        smoothed_flag = False  # flag to check if the image was smoothed already
        filtered = 0

        xi, yi = np.meshgrid(range(self.PanoramaImage.shape[1]), range(self.PanoramaImage.shape[0]))
        xi = np.radians(xi * self.azimuth_spacing)
        yi = np.pi / 2 - np.radians(yi * self.elevation_spacing)

        # Smoothing the range image using adaptive smoothing
        sigma = kwargs.get('sigma', 0)

        if 'smoothing_function' in kwargs:
            if kwargs['smoothin_function'] == 'gaussianBlur':
                filtered = cv2.GaussianBlur(self.PanoramaImage, ksize, sigma)
                smoothed_flag = True
            elif kwargs['smoothing_function'] == None:
                filtered = self.PanoramaImage
                smoothed_flag = True
        if smoothed_flag is False:
            filtered, sigma, ksize = self.adaptive_smoothing(self.PanoramaImage, phenomenaSize)

        # Computing gradient map in the scan after lowpass filter
        import MyTools as mt
        gradient = mt.computeImageGradient(filtered, ksize=ksize, sigma=sigma, gradientType=gradientType)

        # Computing Normal vectors
        # finding the ray direction (Zeibak Thesis: eq. 20, p. 58)
        x = filtered * np.cos(yi) * np.cos(xi)
        y = filtered * np.cos(yi) * np.sin(xi)
        z = filtered * np.sin(yi)

        xyz = PointSet(np.vstack([x.flatten(), y.flatten(), z.flatten()]).T)

        n = NormalsFactory.normalsComputation_in_raster(x, y, z)

        return n, gradient, xyz, filtered

    @property
    def Size(self):
        """
        The image panorama shape

        """
        return self.PanoramaImage.shape

    def adaptive_smoothing(self, panoramaImage, phenomena_size, **kwargs):
        r"""
        Adaptive smoothing of range image according to another (self) panorama image

        .. note::
           The function is implemented similar to other smoothing functions (e.g., `cv2.GaussianBlur`; `cv2.Blur`), so
           different kinds of smoothing functions will be applicable by sending (without `if`).
           This means that the function will return computed `sigma` and `ksize` according to
           the average - and they will be returned in the variable.

         An adaptive smoothing is implemented as a family of convolution kernels characterized by
         different :math:`\sigma` values where:

        .. math::
             d(\rho )=\frac{D}{\rho \Delta }

        with :math: `D`, the object-space size of the window, and :math: `\Delta` the angular sampling resolution
        and :math: `\rho` is the measured range.

        :cite:`Arav2013`

        .. todo::
           Add optionality for adaptive smoothing for other properties panoramas, where adaptation is according to
           range information

        :param phenomena_size: the minimal size of the phenomena that we don't want to smooth (:math: `D` above)
         default: 0.12 m

        :param panoramaImage: the image to smooth

        :return:
           - smoothed image
           - mean sigma size
           - mean kernel size

        :rtype: tuple

        """
        if phenomena_size == 0:
            phenomena_size = 0.2  # in meters
        filtered_image = np.zeros(self.Size)
        sigmas = []
        ksizes = []
        scan_resolution = np.mean([self.azimuth_spacing, self.elevation_spacing])
        for i in range(self.Size[0]):
            for j in range(self.Size[1]):
                rho = self.PanoramaImage[i, j]
                if rho == self.void_data:
                    filtered_image[i, j] = self.void_data
                    continue
                elif np.isnan(rho):
                    filtered_image[i, j] = np.nan
                    continue
                else:
                    # Create the filter and define the window size according to the changing resolution (sigma)
                    current_sigma = 2.5 * phenomena_size / (rho * scan_resolution)
                    sigmas.append(current_sigma)
                    ksize = np.ceil(2 * current_sigma + 1).astype(int)
                    if ksize % 2 == 0:
                        ksize += 1

                    ksizes.append(ksize)

                    gauss_kernel = cv2.getGaussianKernel(ksize, current_sigma)
                    gauss_kernel = gauss_kernel.dot(gauss_kernel.T)

                    # Define the window
                    win_size = (ksize / 2).astype(int)
                    i_start = max(0, i - win_size)
                    i_start_win = max(0, win_size - i - i_start)
                    i_end = min(self.Size[0], i + win_size + 1)
                    i_end_win = min(gauss_kernel.shape[0], gauss_kernel.shape[0] - ((i + win_size) - i_end))

                    j_start = max(0, j - win_size)
                    j_start_win = max(0, win_size - j - j_start)
                    j_end = min(self.Size[1], j + win_size + 1)
                    j_end_win = min(gauss_kernel.shape[1], gauss_kernel.shape[1] - ((j + win_size) - j_end))

                    patch = panoramaImage[i_start:i_end, j_start: j_end]
                    gauss_win = gauss_kernel[i_start_win:i_end_win, j_start_win: j_end_win]

                    non_nan = np.where(patch != self.void_data)
                    filtered_image[i, j] = sum(patch[non_nan] * gauss_win[non_nan]) / sum(gauss_win[non_nan])

        sigma = np.mean(sigmas)
        ksize = np.mean(ksizes)
        return filtered_image, sigma, ksize
