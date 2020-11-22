'''
Created on Nov 25, 2015

@author: Reuma
'''

import cv2
#
# import geopandas as gpd
import h5py
# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.ndimage import filters
# from shapely.geometry import Polygon
from skimage import measure
import scipy.interpolate as interp




def chi2_distance(histA, histB, eps=1e-10):
    r"""
    Compute the :math:`\chi^2` distance between two histograms

    .. math::
        D_{\chi^2}(hist_A, hist_B) = \sum_{n=0}^N \frac{\left(hist_A - hist_B\right)^2}{hist_A + hist_B}

    :param histA: histogram A
    :param histB: histogram B (which A compares to)
    :param eps: an epsilon for non-zero division

    :return: the distance between the histograms

    :rtype: np.ndarray

    """

    if np.all(histA == histB):
        # print('same hist')
        return 0

    return np.sum(np.square(histA - histB) / (histA + histB + eps), axis=1)


def imshow(img, scale_val = 1, ax = None, cmap = 'gray', *args, **kwargs):
    '''
    shows pixel value in the image window
    '''
    if ax is None:
        ax = plt.gca()

    def format_coord(x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return "%s @ [%4i, %4i]" % (img[y, x], x, y)
        except IndexError:
            return ""
    im = ax.imshow(img, cmap = cmap, *args, **kwargs)
    ax.format_coord = format_coord
    ax.figure.canvas.draw()
    return im


def im2double(img):
    '''
    MATLAB equivalent
    @param img: image to convert into double
    @return: numpy array as a double array
    '''
    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    out = (img.astype('float') - min_val) / (max_val - min_val)
    return out


def computeImageGradient(I, **kwargs):
    '''
    Computes the gradient to a given image

    :param I: image to which the gradient is computed
    :param ksize: kernel size, for blurring and derivatives
    :param sigma: sigma for LoG gradient
    :param gradientType: 'L1' L1 norm of grad(I); 'L2' L2-norm of grad(I); 'LoG' Laplacian of gaussian

    :return: an image of the gradient magnitude
    '''

    ksize = int(kwargs.get('ksize', 5))
    gradientType = kwargs.get('gradientType', 'L1')
    sigma = kwargs.get('sigma', 2.5)

    gradient = None

    # img = cv2.GaussianBlur(I, (ksize, ksize), sigma)
    img = I
    # compute image gradient
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = ksize)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = ksize)

    if gradientType == 'L1':
        gradient = cv2.GaussianBlur((np.abs(dx) + np.abs(dy)), (ksize, ksize), sigma)  # L1-norm of grad(I)
    elif gradientType == 'L2':
        gradient = cv2.GaussianBlur(np.sqrt(dx ** 2 + dy ** 2), (ksize, ksize), sigma)
    elif gradientType == 'LoG':
        gradient = cv2.GaussianBlur(filters.gaussian_laplace(I, sigma), (ksize, ksize), sigma)

    # return cv2.normalize((gradient).astype('float'), None, 0.0,1.0, cv2.NORM_MINMAX)
    return gradient

def repartitionCurve(c, dh):
    '''
    Re-partitions the curve according to dh

    :param c: curve
    :param dh: length between points

    :return: the re-partitioned curve
    '''
    dc = np.diff(c, axis = 0)
    dtau = norm(dc, axis = 1)
    tau = np.cumsum(dtau)
    tau = np.append(0, tau)

    interpolator_x = interp1d(tau, c[:, 0])
    interpolator_y = interp1d(tau, c[:, 1])

    tau_new = np.arange(0, tau[-1], dh)
    tau_new = np.append(tau_new, tau[-1])
    return tau_new, interpolator_x(tau_new), interpolator_y(tau_new)


def curveCentralDerivatives(c, repartition = 5):
    '''
    Computes the first and second central derivative of a curve
    :param c: curve points, not necessarily arc-length
    :param repartition: the step for repartition. if negative does not change the curve partition
    :return: tuple, (c', c")
    '''
    dc = np.diff(c, axis = 0)
    dtau = norm(dc, axis = 1)
    dtau_ = np.mean(dtau)

    dtau = np.append(dtau, dtau_)[:, None]

    tau = np.cumsum(dtau)
    # re-partition the curve
    if repartition > 0:
        tau, c_x, c_y = repartitionCurve(c, repartition)
        c = np.vstack((c_x, c_y)).T
        dtau = repartition

    c_extrapolated = np.vstack((c[-1, :], c, c[0, :]))

    # derivatives
    dc = c_extrapolated[2:, :] - c_extrapolated[:-2, :]
    d2c = c_extrapolated[2:, :] + c_extrapolated[:-2, :] - 2 * c

    dc_dtau = dc / (2 * dtau)
    d2c_dtau2 = d2c / dtau ** 2

    return c, dc_dtau, d2c_dtau2


def computeImageDerivatives(img, order, **kwargs):
    """
        Computes image derivatives up to order 2.

        :param img: the image to which the derivatives should be computed
        :param order: order needed (1 or 2)
        :param ksize: filter kernel size (1, 3, 5, or 7)
        :param resolution: kernel resolution
        :param sigma: sigma for gaussian blurring. Default: 1. If sigma=0 no smoothing is carried out
        :param window: tuple of window size for blurring


        :return: tuple of the derivatives in the following order: (dx, dy, dxx, dyy, dxy)
    """

    params = {'order': 1,
              'ksize': 3,
              'resolution': 1,
              'sigma': 1.,
              'window': (0,0)}
    params.update(kwargs)
    ksize = np.int(params['ksize'])

    img = np.float64(img)

    img_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    img_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

    if params['sigma'] != 0:
        img_x = cv2.GaussianBlur(img_x, params['window'], params['sigma'])
        img_y = cv2.GaussianBlur(img_y, params['window'], params['sigma'])

    # img_x = cv2.normalize(img_x.astype('float'), None, 0.0, 1.0,
    #                       cv2.NORM_MINMAX)
    # img_y = cv2.normalize(img_y.astype('float'), None, 0.0, 1.0,
    #                       cv2.NORM_MINMAX)

    if order == 2:
        img_xx = cv2.Sobel(img_x, cv2.CV_64F, 1, 0, ksize = ksize)
        img_yy = cv2.Sobel(img_y, cv2.CV_64F, 0, 1, ksize = ksize)
        img_xy = cv2.Sobel(img_x, cv2.CV_64F, 0, 1, ksize = ksize)

        if params['sigma'] !=0:
            img_xx = cv2.GaussianBlur(img_xx, params['window'], params['sigma'])
            img_yy = cv2.GaussianBlur(img_yy, params['window'], params['sigma'])
            img_xy = cv2.GaussianBlur(img_xy, params['window'], params['sigma'])

        # img_xx = cv2.normalize(img_xx.astype('float'), None, 0.0, 1.0,
        #                        cv2.NORM_MINMAX)
        # img_yy = cv2.normalize(img_yy.astype('float'), None, 0.0, 1.0,
        #                        cv2.NORM_MINMAX)
        # img_xy = cv2.normalize(img_xy.astype('float'), None, 0.0, 1.0,
        #                        cv2.NORM_MINMAX)

        return img_x, img_y, img_xx, img_yy, img_xy
    else:
        return img_x, img_y

def computeImageGradient_numeric(img, gradientType='L1', ksize=3, sigma=1, resolution=1, blur_window=(0,0), **kwargs):
    """

        :param img: the image to which the derivatives should be computed
        :param gradientType: 'L1' L1 norm of grad(I); 'L2' L2-norm of grad(I); 'LoG' Laplacian of gaussian
        :param ksize: size of the differentiation window
        :param resolution: kernel resolution
        :param sigma: sigma for gaussian blurring. Default: 1. If sigma=0 no smoothing is carried out
        :param blur_window: tuple of window size for blurring

        :type img: np.array
        :type gradientType: str
        :type ksize: int
        :type resolution: float
        :type sigma: float
        :type blur_window: tuple

        :return: an image of the gradient magnitude
        :rtype: np.array
    """

    gradient = None

    # img = cv2.GaussianBlur(I, (ksize, ksize), sigma)

    # compute image gradient (numeric)
    dx, dy = computeImageDerivatives_numeric(img, 1, ksize, sigma, resolution,blur_window)

    if gradientType == 'L1':
        gradient = cv2.GaussianBlur((np.abs(dx) + np.abs(dy)), (ksize, ksize), sigma)  # L1-norm of grad(I)
    elif gradientType == 'L2':
        gradient = cv2.GaussianBlur(np.sqrt(dx ** 2 + dy ** 2), (ksize, ksize), sigma)
    elif gradientType == 'LoG':
        gradient = filters.gaussian_laplace(img, sigma)

    # return cv2.normalize((gradient).astype('float'), None, 0.0,1.0, cv2.NORM_MINMAX)
    return gradient

def computeImageDerivatives_numeric(img, order, ksize=3, sigma=1., resolution=1., blur_window=(0,0), **kwargs):
    """
        Computes numeric image derivatives up to order 2.

        :param img: the image to which the derivatives should be computed
        :param order: order needed (1 or 2)
        :param ksize: size of the differentiation window
        :param resolution: kernel resolution
        :param sigma: sigma for gaussian blurring. Default: 1. If sigma=0 no smoothing is carried out
        :param blur_window: tuple of window size for blurring

        :type img: np.array
        :type order: int
        :type ksize: int
        :type resolution: float
        :type sigma: float
        :type blur_window: tuple


        :return: tuple of the derivatives in the following order: (dx, dy, dxx, dyy, dxy)

        .. note::
            Eq. numbers from Amit Baruch Dissertation
    """

    img = (img).astype('float64')

    # if blurring is required before differentiation
    if sigma != 0:
        img = cv2.GaussianBlur(img, blur_window, sigma)

    floord = int(ksize)

    # Derivatives (eq. 3-37)

    # x direction
    I1 = np.hstack((img[: , floord: ], img[:, - floord:]))
    I2 = np.hstack((img[:, : floord], img[:, : - floord]))
    Zx = (I1 - I2) / (2 * resolution * np.ceil(ksize))

    # y direction
    I3 = np.vstack((img[floord: , :], img[-floord:, :]))
    I4 = np.vstack((img[: floord, :], img[: -floord, :]))
    Zy = (I3 - I4) / (2 * resolution * ksize)

    # second order derivatives
    if order == 2:
        I5a = np.hstack((img[: - floord, : floord], img[: - floord, : - floord]))
        I5 = np.vstack((I5a[: floord, :], I5a))
        I6a = np.hstack((img[floord:, floord:], img[floord:, - floord:]))
        I6 = np.vstack((I6a, I6a[- floord:, :]))

        I7a = np.hstack((img[floord:, : floord], img[floord:, :-floord]))
        I7 = np.vstack((I7a, I7a[- floord:, :]))
        I8a = np.hstack((img[floord:, floord:], img[floord:, - floord:]))
        I8 = np.vstack((I8a, I8a[- floord:, :]))

        # eq. (3-40)
        Zxx = (-2 * img + I1 + I2) / (ksize * resolution) ** 2
        Zyy = (-2 * img + I3 + I4) / (ksize * resolution) ** 2
        Zxy = (I7 + I6 - I5 - I8) / (2 * ksize * resolution) ** 2

        return Zx, Zy, Zxx, Zyy, Zxy

    return Zx, Zy


def DoG_filter(image, **kwargs):
    """
    Compute the difference of gaussians filter.
    :param image: the image
    :param sigma1, sigma2: the sigma to be used for blurring. cv2 will decide the size accordingly.
    :param ksize1, ksize2: the size of the filter to be used for blurring. cv2 will decide the sigma accordingly.
    :return: filtered image
    """

    filter_params = {'sigma1': 0,
                     'sigma2': 0,
                     'ksize1': 0,
                     'ksize2': 0}
    filter_params.update(kwargs)

    # large kernel minus smaller one
    if filter_params['sigma1'] > filter_params['sigma2']:
        sigma1 = filter_params['sigma2']
        sigma2 = filter_params['sigma1']
    else:
        sigma1 = filter_params['sigma1']
        sigma2 = filter_params['sigma2']

    if filter_params['ksize1'] > filter_params['ksize2']:
        ksize1 = (filter_params['ksize2'],filter_params['ksize2'])
        ksize2 = (filter_params['ksize1'], filter_params['ksize1'])
    else:
        ksize1 = (filter_params['ksize1'], filter_params['ksize1'])
        ksize2 = (filter_params['ksize2'], filter_params['ksize2'])

    blur1 = cv2.GaussianBlur(image, ksize1, sigma1)
    blur2 = cv2.GaussianBlur(image, ksize2, sigma2)

    return blur2-blur1


def chooseLargestContours(contours, labelProp, minArea):
    '''
    leaves only contours with area larger than minArea
    :param contours: list of contours
    :param labelProp: properties of labeled area
    :param minArea: minimal area needed
    :return: list of "large" contours
    '''
    contours_filtered = []

    for prop, c in zip(labelProp, contours):
        if prop.area >= minArea and prop.area <= 10e6:
            contours_filtered.append(c)
    return contours_filtered

def is_pos_semidef(x):
    """
    check if a matrix is positive semidefinite
    :param x: the matrix
    :return:  Boolean: True if it is, False is not.
    """
    return np.all(np.linalg.eigvals(x) >= 0)


def intersect2d(a, b, **kwargs):
    """
    finds the recurring elements in two arrays (row-wise)

    :param axis: column-wise
    :return:
    """
    axis = kwargs.get('axis', 0)
    nrows, ncols = a.shape
    if axis == 0:
        byaxis = ncols

    else:
        # TODO: CHECK THIS ONE WROKS, DIDN'T TRY THAT
        byaxis = nrows
    dtype = {'names': ['f{}'.format(i) for i in range(byaxis)],
             'formats': byaxis * [a.dtype]}
    aa = a.copy()
    bb = b.copy()
    c = np.intersect1d(aa.view(dtype), bb.view(dtype))
    return c.view(a.dtype).reshape(-1, byaxis)


def CreateFilename(filename, mode='w', **kwargs):
    """
   Checks or creates a file object according to specifications given.
       Default is hdf5.

   .. warning:: Need to be implemented for other formats than hdf5

   :param filename: can be a filename or a path and filename, with or without extension.
   :param mode: is an optional string that specifies the mode in which the file is opened.
    It defaults to 'w' which means open for writing in text mode.

   :param extension: 'h5', 'json', 'shp', 'pts', etc...

   :type filename: str
   :type mode: str
   :type extension: str

   :return: a file object (according to the extension) and its extension

   """

    import re

    matched = re.match('(.*)\.([a-z].*)', filename)

    if matched is None:
        # if no extension is in filename, add
        extension = kwargs.get('extension', 'h5')  # if no extension given - default is h5

        filename = filename + '.' + extension
    else:
        # otherwise - use the extension in filename
        extension = matched.group(2)

    if extension == 'h5':
        return (h5py.File(filename, mode), extension)

    else:  # change if needed
        return (open(filename, mode), extension)


def draw_contours(func, ax, img, hold=False, blob_size=5, color_random=False,  **kwargs):
    """
    Draws the contours of a specific iteration

    :param func: the function which contours should be drawn
    :param ax: the axes to draw upon
    :param image: the image on which the contours will be drawn
    :param hold: erase image from previous drawings or not. Default: False
    :param color: the color which the contour will be drawn. Default: random for each curve (send True)
    :param linewidth: the line width of the curves. Default: 1
    :param blob_size: the minimal area of a contour to draw

    :type func: np.ndarray
    :type ax: plt.axes
    :type hold: bool
    :type color: str or bool


    :return: the figure

    """
    import random

    if not hold:
        ax.cla()
    # ax.axis("off")

    ax.set_ylim([img.shape[0], 0])
    ax.set_xlim([0, img.shape[1]])

    color = kwargs.get('color', 'b')
    linewidth = kwargs.get('linewidth', 1)

    tmp_func = func.copy()
    tmp_func[np.abs(tmp_func)<=1e-4] = 1e-4
    # segmenting and presenting the found areas
    # function_binary[np.where(func > 0.)] = 0
    # function_binary[np.where(func <= 0.)] = 1
    # function_binary = np.uint8(function_binary)

    contours = measure.find_contours(np.flipud(tmp_func), 0.)
    # blob_labels = measure.label(function_binary, background=1)
    # label_props = measure.regionprops(blob_labels)`
    # 
    # contours = chooseLargestContours(contours, label_props, blob_size)
    if not hold:
        imshow(np.flipud(img))
    l_curve = []

    for c in contours:
        # make sure area is not too small and not too big
        x = c[:, 0]
        y = c[:, 1]
        area = 0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y))
        area = np.abs(area)
        if area <= blob_size or area >= 10e6:
            continue

        c[:, [0, 1]] = c[:, [1, 0]]  # swapping between columns: x will be at the first column and y on the second

        if color_random:
            color = (random.random(), random.random(), random.random())

        curve, = ax.plot(c[:, 0], c[:, 1], '-', color=color, linewidth=linewidth)

        l_curve.append(curve)

    return l_curve, ax

def connected_components_filtering(image, min_area=None, connectivity=8):
    """
    Filter an image using connected components according to maximal area size

    :param image: the image ro filter
    :param max_area: the minimum allowed area. If None, it will be the median area of the CC.
    :param connectivity: the connectivity type (4/8). Default: 8
    :type image: np.array
    :type max_area: int

    :return: filtered image

    :rtype: np.array
    """

    from scipy import ndimage

    label_img, cc_num = ndimage.label(image)
    CC = ndimage.find_objects(label_img)
    cc_areas = ndimage.sum(image, label_img, range(cc_num + 1))

    if min_area is None:
        min_area = np.percentile(cc_areas, 50)

    area_mask = (cc_areas < min_area)
    label_img[area_mask[label_img]] = 0

    return label_img, CC

def curve2D_toGeoSeries(curve):
    """
    Transforms a 2D line to a geopandas GeoSeries

    :param curve: pyplot.2Dline

    :return: geopandas GeoSeries of a polygon
    """

    xy = curve.get_path().vertices
    if np.all(xy[-1, :] == xy[0, :]):
        xy = xy[1:, :]
    polygon = Polygon(xy)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def curves2D_toGeoDataframe(curves):
    """
    Transforms multiple 2D lines to geopandas GeoDataFrame
    :param curves: list of pyplot.lines.Line2D

    :type curves: list

    :return: geodataframe of the polygons

    :rtype: gpd.GeoDataFrame

    """

    geometry = [curve2D_toGeoSeries(curve) for curve in curves]

    df = pd.DataFrame({'id': range(len(geometry)), 'coordinates': geometry})
    return gpd.GeoDataFrame(df, geometry='coordinates')


def eig(matrix):
    """
    Computes eigenvalues and eigenvectors and returns them sorted from smallest eigenvalues to largest

    :param matrix: matrix to compute the eigenvalues and eigenvectors

    :type matrix: nd-array

    :return eigenValues, eigenVectors: sorted from small to large
    """
    eigVals, eigVectors = np.linalg.eig(matrix)
    sort_perm = eigVals.argsort()
    eigVals.sort()  # <-- This sorts the list in place.
    eigVectors = eigVectors[:, sort_perm]
    return eigVals, eigVectors


def ballTree_saliency(pointset, scale, neighborhood_properties, curvature_attribute, leaf_size=10,
                      weight_distance=1, weight_normals=1, verbose=False):
    """
    Compute curvature based saliency using ball-tree and k-nearest-neighbors.

    The saliency for every set of points within a ball-tree cell is computed as one for all points.

    .. seealso::
        :meth:`SaliencyFactory.curvature_saliency`

    :param pointset: a point cloud of any sort
    :param scale: the scale of the minimal phenomena (size of the smallest ball-tree cell)
    :param neighborhood_properties: k nearest neighbors
    :param curvature_attribute: the attribute according to which the curvature is measured.

    :param leaf_size: the minimum number of points in the ball tree leaves. Default: 10
    :param weight_distance: weights for distance element. Default: 1.
    :param weight_normals: weights for normal element. Default: 1.
    :param verbose: print running messages. Default: False

    :type pointset: np.ndarray, PointSet.PointSet, BallTreePointSet.BallTreePointSet, PointSetOpen3D.PointSetOpen3D
    :type scale: float
    :type neighborhood_properties: int or float
    :type leaf_size: int
    :type weight_normals: float
    :type weight_distance: float
    :type verbose: bool

    :return: curvature property, normals property, saliency property

    :rtype: CurvatureProperty, NormalsProperty, SaliencyProperty
    """

    from CurvatureFactory import CurvatureFactory, CurvatureProperty
    from SaliencyFactory import SaliencyFactory, SaliencyProperty
    from NeighborsFactory import NeighborsFactory
    from BallTreePointSet import BallTreePointSet
    from PointSubSet import PointSubSet
    from TensorFactory import TensorFactory, TensorProperty
    from NormalsFactory import NormalsFactory, NormalsProperty

    # 1. Build the BallTree
    if isinstance(pointset, BallTreePointSet):
        balltree = pointset

    else:
        balltree = BallTreePointSet(pointset, leaf_size=leaf_size)

    # 2. Choose vertices according to minimal scale
    nodes = balltree.getSmallestNodesOfSize(scale, 'leaves')

    # 3. Build tensors out of each node's points, and build their neighborhoods
    tensors = []
    new_cloud = []
    if verbose:
        i = 0

    for node in nodes:
        if verbose:
            i += 1
            if i == 1192:
                print('hello')
        tmp_subset = PointSubSet(pointset, balltree.getPointsOfNode(node))
        tensors.append(TensorFactory.tensorFromPoints(tmp_subset, -1))
        new_cloud.append(tensors[-1].reference_point)

    new_bt = BallTreePointSet(np.asarray(new_cloud))  # the new cloud of the centers of mass of each tensor
    tensors_property = TensorProperty(new_bt, np.asarray(tensors))
    neighbors = NeighborsFactory.balltreePointSet_knn(new_bt, neighborhood_properties)  # neighborhood construction

    # 4. Curvature for each tensor
    tensor_curvature = CurvatureFactory.tensorProperty_3parameters(tensors_property)

    # 5. Normal for each tensor
    tensor_normals = NormalsFactory.normals_from_tensors(tensors_property)

    # 6.  Saliency for each tensor
    tensor_saliency = SaliencyFactory.directional_saliency(neighbors, tensor_normals, tensor_curvature,
                                                         curvature_attribute=curvature_attribute, verbose=verbose)

    # 7. Assign saliency value for each point in the tensor
    pcl_saliency = SaliencyProperty(pointset)
    pcl_curvature = CurvatureProperty(pointset)
    pcl_normals = NormalsProperty(pointset)

    for node_, saliency, curvature, normal in zip(nodes, tensor_saliency, tensor_curvature, tensor_normals):
        idx = balltree.getPointsOfNode(node_)
        pcl_saliency.setPointSaliency(idx, saliency)
        pcl_curvature.setPointCurvature(idx, curvature)
        pcl_normals.setPointNormal(idx, normal)

    return pcl_saliency, pcl_curvature, pcl_normals

def chunking_dot(big_matrix, small_matrix, chunk_size=100):
    # Make a copy if the array is not already contiguous
    small_matrix = np.ascontiguousarray(small_matrix)
    R = np.empty((big_matrix.shape[0], small_matrix.shape[1]))
    for i in range(0, R.shape[0], chunk_size):
        end = i + chunk_size
        R[i:end] = np.dot(big_matrix[i:end], small_matrix)
    return R


def scale_values(array, min=0., max=1.):
    r"""
    Scale values between min and max range

    .. math::
        f_{new} = \frac{(max-min) (f-f_{min})}{f_{max}-f_{min}} + min

    :param array: the array that will be re-scaled
    :param min: the minimal value of the range (default: 0)
    :param max: the maximal value of the range (default: 1)

    :type array: np.array
    :type min: float
    :type max: float

    :return: rescaled array

    :rtype: np.array
    """
    if len(array) <1:
        return array
    return (max-min) * (array - array.min()) / (array.max() - array.min()) + min

def make_zero(array, eps=1e-5):
    """
    Convert numerical zero to zero

    :param array: an array with various parameters
    :param eps: a thershold below which the number is zero (default: 1e-5

    :type array: np.array
    :type eps: float

    :return: the same array with zeros instead of :math:`|x|<eps`

    :rtype: np.array

    """
    array[np.abs(array)< eps] = 0
    return array

def interpolate_data(data, attribute_data, resolution, func=interp.NearestNDInterpolator, **kwargs):
    """
    Interpolate data according to a specific attribute

    :param data: the table to interpolate. Should have at least x,y and an attribute to interpolate (z or other)
    :param attribute_data: values to interpolate
    :param func: the function with which the data will be interpolated. Default: nearest neighbors
    :param kwargs: dictionary for function's arguments

    :type data: np.ndarray
    :type attribute: int
    :type func: function

    :return: interpolated data
    """
    x = np.arange(data[:,0].min(), data[:,0].max(), resolution)
    y = np.arange(data[:, 1].min(), data[:, 1].max(), resolution)
    xx, yy = np.meshgrid(x,y)
    interped = func(data[:,:2], attribute_data, **kwargs)
    zz = interped((xx, yy)).reshape(xx.shape)

    return zz


def size_filter(binary_map, min_size=100):
    """
    Filters out small sized elements in a binary image (called size filter according to Sagi)

    :param binary_map: the map to work on
    :param min_size: the minimal size of an element to be left

    :type binary_map: np.array
    :type min_size: int
    :return: the clean image

    :rtype: np.array

    *source* `https://stackoverflow.com/questions/57283802/remove-small-whits-dots-from-binary-image-using-opencv-python`_
    """

    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

    # get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:,cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= min_size:   #keep
            result[labels == i + 1] = 255

    return result


def non_maxima_suppression(img):
    """
    Computes non-maxima suppression on an image  

    :param img: the image for non-maxima suppression

    :return: the image after non maxima suppression
    """

    dx, dy = computeImageDerivatives(img, 1)
    phase = np.arctan2(dx, dy)

    gmax = np.zeros(img.shape)

    phase[phase < 0] += 360
    for i in np.arange(1, gmax.shape[0]-1):
        for j in np.arange(1, gmax.shape[1]-1):

            # 0 degrees
            if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (phase[i][j] >= 157.5 and phase[i][j] < 202.5):
              if img[i][j] >= img[i][j + 1] and img[i][j] >= img[i][j - 1]:
                gmax[i][j] = img[i][j]
            # 45 degrees
            if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):
              if img[i][j] >= img[i - 1][j + 1] and img[i][j] >= img[i + 1][j - 1]:
                gmax[i][j] = img[i][j]
            # 90 degrees
            if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):
              if img[i][j] >= img[i - 1][j] and img[i][j] >= img[i + 1][j]:
                gmax[i][j] = img[i][j]
            # 135 degrees
            if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):
              if img[i][j] >= img[i - 1][j - 1] and img[i][j] >= img[i + 1][j + 1]:
                gmax[i][j] = img[i][j]
    return gmax


if __name__ == '__main__':
    img_orig = cv2.cvtColor(cv2.imread(r'/home/reuma/ownCloud/Data/Images/channel91.png'), cv2.COLOR_BGR2GRAY)
    img_normed = cv2.normalize(img_orig.astype('float'), None, 0.0, 1.0,
                               cv2.NORM_MINMAX)  # Convert to normalized floating point

    xx, yy = computeImageDerivatives(img_normed, 1, ksize = 5)
    norm_nabla = computeImageGradient(img_normed, gradientType='L2')
    diff = np.sqrt(xx**2 + yy**2) - norm_nabla

    print(diff)
