import cv2
import numpy as np
from numpy import ones

from DataClasses.PointSet import PointSet
from DataClasses.PointSubSet import PointSubSet
from Properties.BaseProperty import BaseProperty
from Properties.Normals.NormalsFactory import NormalsFactory
from Properties.Transformations.SphericalCoordinatesProperty import SphericalCoordinatesProperty


class PanoramaProperty(BaseProperty):
    """
    A panoramic representation of the point set.

    .. note:: For the Scanstation C10 the measurements' angular resolution for both elevation and azimuth directions:

            * Low: :math: 0.2^\circ
            * Medium: :math: 0.11^\circ
            * High: :math: 0.057^\circ
            * Highest: :math: 0.028^\circ

    The panorama will always include a range image and will include intensity image data if available, and other properties arranges as panorama, if available.
    """

    # data stored as images
    __panoramaData = None  # A m-by-n-by-p array in which the panorama is stored. Can be either range or property
    __rangeData = None  # A m-by-n-by-p array in which the range data is stored.
    __intensityData = None  # A m-by-n-by-p array in which the intensity data is stored
    __panoramaIndex = None  # A m-by-n-by-p array in which the indices of the points are stored

    # scan properties
    __voidData = 250  # A number indicating missing data in the panorama
    __minAzimuth = 0  # The minimal azimuth value
    __maxAzimuth = 360  # The maximal azimuth value
    __minElevation = -45  # The minimal elevation angle value
    __maxElevation = 90  # The maximal elevation angle value
    __azimuthSpacing = 0.057  # The spacing between points in the azimuth direction
    __elevationSpacing = 0.057  # The spacing between points in the elevation angle direction

    __rowIndexes = None  # An array of indexes corresponding to the row number to which each point belongs to
    __columnIndexes = None  # An array of indexes corresponding to the column number to which each point belongs to

    __sphericalCoordinates = None # the points coordinate in spheric coordinates

    def __init__(self, sphericalCoordinates, rowIndexes=None, columnIndexes=None, panoramaData=None, intensityData = None, **kwargs):
        """
        Constuctor - Creates a panoramic view of the data sent

        :param sphericalCoordinates: SphericalCoordinates property
        :param panoramaData: data to be represented as a panorama (e.g. range, intensity, etc.). Default: range
        :param rowIndexes: row indices of the points in the point set based on the elevation angles
        :param columnIndexes: column indices of the points in the point set based on the azimuth angles


        :type sphericalCoordinates: SphericalCoordinatesProperty
        :type rowIndexes: np.array
        :type columnIndexes: np.array
        :type panoramaData: np.array

        """
        super(PanoramaProperty, self).__init__(sphericalCoordinates.Points)

        self.load(**kwargs)

        self.__columnIndexes = columnIndexes
        self.__rowIndexes = rowIndexes
        self.__sphericalCoordinates = sphericalCoordinates

        numRows = rowIndexes.max() + 1
        numColumns = columnIndexes.max() + 1

        self.__rangeData =  self.void_data * ones((numRows, numColumns))
        self.__rangeData[rowIndexes, columnIndexes] = sphericalCoordinates.ranges

        self.__panoramaIndex = ones((numRows, numColumns)) * np.NaN
        self.__panoramaIndex[rowIndexes, columnIndexes] = np.arange(0, self.sphericalCoordinates.Size, dtype=np.int)

        if intensityData is not None:
            self.__intensityData = self.void_data * ones((numRows, numColumns))
            self.__intensityData[rowIndexes, columnIndexes] = intensityData

        if panoramaData is not None:
            if len(panoramaData.shape) == 1:
                self.__panoramaData = self.void_data * ones((numRows, numColumns))
                self.__panoramaData[rowIndexes, columnIndexes] = panoramaData
            else:
                self.__panoramaData = self.void_data * ones((numRows, numColumns, panoramaData.shape[1]))
                self.__panoramaData[rowIndexes, columnIndexes, :] = panoramaData[:, :]
    @property
    def panoramaIndex(self):
        """
        The point indices in a panorama. If no point exist the value is NaN

        :return:
        """
        panoramaIndex = self.__panoramaIndex.copy()
        panoramaIndex[np.isnan(self.__panoramaIndex)] = -1

        return panoramaIndex.astype('int')

    @property
    def sphericalCoordinates(self):
        """
        Point in spherical coordinates

        """
        return self.__sphericalCoordinates

    @property
    def panoramaImage(self):
        """
        The data as panorama image. If no data stored, the range image is returned

        """
        if self.__panoramaData is None:
            return self.__rangeData
        else:
            return self.__panoramaData

    @property
    def rangeImage(self):
        """
        The range as panorama

        """
        return self.__rangeData

    @property
    def intensityImage(self):
        """
        The intensity data as panorama image. If no data stored, it will return None.

        """

        return self.__intensityData

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

    @property
    def void_data(self):
        """
        Value set for void areas

        """
        return self.__voidData

    @property
    def Size(self):
        """
        The image panorama shape

        """
        return self.rangeImage.shape

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

    def pano2rad(self):
        """
        Convert the axes of the panorama to their value as radians

        :return: azimuth and elevation angles of the panorama

        :rtype: tuple
        """

        xi, yi = np.meshgrid(range(self.rangeImage.shape[1]), range(self.rangeImage.shape[0]))
        xi = np.radians(xi * self.azimuth_spacing)
        yi = np.pi / 2 - np.radians(yi * self.elevation_spacing)

        return xi, yi

    def adaptive_smoothing(self, phenomena_size, img_to_smooth=None, verbose=False, **kwargs):
        r"""
        Adaptive smoothing of the panorama image according to the range image.

        .. note::
           The function is implemented similar to other smoothing functions (e.g., `cv2.GaussianBlur`; `cv2.Blur`), so
           different kinds of smoothing functions will be applicable by sending (without `if`).
           This means that the function will return computed `sigma` and `ksize` according to
           the average - and they will be returned in the variable.

         An adaptive smoothing is implemented as a family of convolution kernels characterized by
         different :math:`\sigma` values where:

        .. math::
             d(\rho )=\frac{D}{\rho \Delta }

        with :math:`D`, the object-space size of the window, and :math:`\Delta` the angular sampling resolution
        and :math:`\rho` is the measured range.

        :cite:`Arav2013`

        .. todo::
           Add optionality for adaptive smoothing for other properties panoramas, where adaptation is according to
           range information

        :param phenomena_size: the minimal size of the phenomena that we don't want to smooth (:math: `D` above)
         default: 0.2 m (and in cases that it is sent 0)
        :param img_to_smooth: the image that should be smoothed.
        :param verbose: print kmean and sigma mean


        :type phenomena_size: float
        :type img_to_smooth: np.ndarray
        :type verbose: bool

        :return: smoothed image


        :rtype: tuple

        """
        import cv2
        from tqdm import trange

        if img_to_smooth is None:
            img_to_smooth = self.rangeImage

        if phenomena_size == 0:
            phenomena_size = 0.2  # in meters
        filtered_image = np.zeros(self.Size)
        sigmas = []
        ksizes = []
        scan_resolution = np.mean([self.azimuth_spacing, self.elevation_spacing])
        for i in trange(self.Size[0], desc='preforming adaptive smoothing'):
            for j in range(self.Size[1]):
                rho = self.rangeImage[i, j]
                if rho == self.void_data:
                    filtered_image[i, j] = self.void_data
                    continue
                elif np.isnan(rho):
                    filtered_image[i, j] = np.nan
                    continue
                else:
                    # Create the filter and define the window size according to the changing resolution (sigma)
                    current_sigma = 2.5 * phenomena_size / (rho * scan_resolution)
                    ksize = np.ceil(2 * current_sigma + 1).astype(int)
                    if ksize % 2 == 0:
                        ksize += 1

                    ksizes.append(ksize)
                    sigmas.append(current_sigma)

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

                    patch = img_to_smooth[i_start:i_end, j_start: j_end]
                    gauss_win = gauss_kernel[i_start_win:i_end_win, j_start_win: j_end_win]

                    non_nan = np.where(patch != self.void_data)
                    filtered_image[i, j] = sum(patch[non_nan] * gauss_win[non_nan]) / sum(gauss_win[non_nan])
        if verbose:
            print('sigma mean {}'.format(np.mean(sigmas)))
            print('ksize mean {}'.format(np.mean(ksizes)))
        return filtered_image

    def computePanoramaDerivatives_adaptive(self, order, ksize=0.2, sigma=1.,img_to_compute='panoramaData',    blur_window=(0,0) ):
        r"""
            Numerical and adaptive computation of central derivatives in x and y directions

            :param order: the derivatives order (1 or 2)
            :param ksize: world window size (meters). Default: 0.2 m
            :param sigma: sigma for gaussian blurring. Default: 1. If sigma=0 no smoothing is carried out
            :param img_to_compute: if sent, the derivatives will be computed on this image, according to the range image. Default: 'panoramaData'
            String possibilities:

            - 'intensity'
            - 'range'

            :param blur_window: tuple of window size for blurring

            :type img: np.array
            :type order: int
            :type ksize: float
            :type img_to_compute: str or np.ndarray
            :type resolution: float
            :type sigma: float
            :type blur_window: tuple

            :return: tuple of the derivatives in the following order: :math:`(d_{\theta}, d_{\phi}, d__{\theta\theta}, d{\phi\phi}, d_{\theta\phi})`
            :rtype: tuple
            """
        from PanoramaUtils import computePanoramaDerivatives_adaptive

        resolution = np.radians(self.azimuth_spacing + self.elevation_spacing) / 2

        if not isinstance(img_to_compute, np.ndarray):

            if img_to_compute == 'panoramaData':
                img_to_compute = self.panoramaImage
            elif img_to_compute == 'intensity':
                img_to_compute = self.intensityImage
            elif img_to_compute == 'range':
                img_to_compute = self.rangeImage

        dx, dy = computePanoramaDerivatives_adaptive(img_to_compute, ksize=ksize, sigma=sigma, resolution=resolution, blur_window=blur_window, rangeImage=self.rangeImage)

        if sigma != 0:
            dx= self.adaptive_smoothing(ksize, dx)
            dy= self.adaptive_smoothing(ksize, dy)

        if order == 2:
            dxx, dxy = computePanoramaDerivatives_adaptive(dx,ksize=ksize, sigma=sigma, resolution=resolution, blur_window=blur_window, rangeImage=self.rangeImage)
            dyy, dyx = computePanoramaDerivatives_adaptive(dy, ksize=ksize, sigma=sigma, resolution=resolution, blur_window=blur_window, rangeImage=self.rangeImage)

            if sigma != 0:
                dxx = self.adaptive_smoothing(ksize, dxx)
                dyy = self.adaptive_smoothing(ksize, dyy)
                dxy = self.adaptive_smoothing(ksize, dxy)

            return dx, dy, dxx, dyy, dxy
        else:
            return dx, dy

    def computePanoramaGradient_adaptive(self, gradientType='L1', ksize=0.2, sigma=1, img_to_compute='panoramaData', blur_window=(0, 0), **kwargs):
        """

        Compute panorama gradient adaptively and numerically

        :param gradientType: 'L1' L1 norm of grad(I); 'L2' L2-norm of grad(I); 'LoG' Laplacian of gaussian
        :param ksize: world window size (meters). Default: 0.2 m
        :param resolution: kernel resolution
        :param sigma: sigma for gaussian blurring. Default: 1. If sigma=0 no smoothing is carried out
        :param img_to_compute: if sent, the derivatives will be computed on this image, according to the range image. Default: 'panoramaData'
        :param blur_window: tuple of window size for blurring

        :type gradientType: str
        :type ksize: float
        :type resolution: float
        :type sigma: float
        :type img_to_compute: str or np.ndarray
        :type blur_window: tuple


        :return: an image of the gradient magnitude
        :rtype: np.array
        """
        from scipy.ndimage import filters

        gradient = None
        if not isinstance(img_to_compute, np.ndarray):
            if img_to_compute == 'panoramaData':
                img_to_compute = self.panoramaImage
            elif img_to_compute == 'intensity':
                img_to_compute = self.intensityImage
            elif img_to_compute == 'range':
                img_to_compute = self.rangeImage

        # compute image gradient (numeric)
        dx, dy = self.computePanoramaDerivatives_adaptive(1, ksize, sigma,  img_to_compute, blur_window)

        if gradientType == 'L1':
            gradient = self.adaptive_smoothing(ksize, (np.abs(dx) + np.abs(dy)))  # L1-norm of grad(I)
        elif gradientType == 'L2':
            gradient = self.adaptive_smoothing(ksize, np.sqrt(dx ** 2 + dy ** 2))
        elif gradientType == 'LoG':
            gradient = filters.gaussian_laplace(self.panoramaImage, sigma)

        # return cv2.normalize((gradient).astype('float'), None, 0.0,1.0, cv2.NORM_MINMAX)
        return gradient