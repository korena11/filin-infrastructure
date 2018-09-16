'''
Created on Nov 25, 2015

@author: Reuma
'''
import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.ndimage import filters
import h5py
from skimage import measure
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon


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

    img = cv2.GaussianBlur(I, (ksize, ksize), sigma)

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
        Computes image derivatives up to order 2. Normalizes the derivatives.

        :param img: the image to which the derivatives should be computed
        :param order: order needed (1 or 2)
        :param ksize: filter kernel size (3,and up)
        :param resolution: kernel resolution
        :param sigma: sigma for gaussian blurring
        :param window: tuple of window size for blurring

        :return: tuple of the derivatives in the following order: (dx, dy, dxx, dyy, dxy)
    """

    params = {'order': 1,
              'ksize': 3,
              'resolution': 1,
              'sigma': 1.,
              'window': (0,0)}
    params.update(kwargs)
    ksize = params['ksize']

    img = np.float64(img)

    img_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = ksize)
    img_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = ksize)

    img_x = cv2.GaussianBlur(img_x, params['window'], params['sigma'])
    img_y = cv2.GaussianBlur(img_y, params['window'], params['sigma'])

    img_x = cv2.normalize(img_x.astype('float'), None, 0.0, 1.0,
                          cv2.NORM_MINMAX)
    img_y = cv2.normalize(img_y.astype('float'), None, 0.0, 1.0,
                          cv2.NORM_MINMAX)

    if order == 2:
        img_xx = cv2.Sobel(img_x, cv2.CV_64F, 1, 0, ksize = ksize)
        img_yy = cv2.Sobel(img_y, cv2.CV_64F, 0, 1, ksize = ksize)
        img_xy = cv2.Sobel(img_x, cv2.CV_64F, 0, 1, ksize = ksize)

        img_xx = cv2.GaussianBlur(img_xx, params['window'], params['sigma'])
        img_yy = cv2.GaussianBlur(img_yy, params['window'], params['sigma'])
        img_xy = cv2.GaussianBlur(img_xy, params['window'], params['sigma'])

        img_xx = cv2.normalize(img_xx.astype('float'), None, 0.0, 1.0,
                               cv2.NORM_MINMAX)
        img_yy = cv2.normalize(img_yy.astype('float'), None, 0.0, 1.0,
                               cv2.NORM_MINMAX)
        img_xy = cv2.normalize(img_xy.astype('float'), None, 0.0, 1.0,
                               cv2.NORM_MINMAX)

        return img_x, img_y, img_xx, img_yy, img_xy
    else:
        return img_x, img_y



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


def CreateFilename(filename, mode = 'w', **kwargs):
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


def draw_contours(func, ax, img, hold = False, **kwargs):
    """
    Draws the contours of a specific iteration

    :param func: the function which contours should be drawn
    :param ax: the axes to draw upon
    :param image: the image on which the contours will be drawn
    :param hold: erase image from previous drawings or not. Default: False
    :param color: the color which the contour will be drawn. Default: random for each curve (send True)

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

    color = kwargs.get('color', True)

    function_binary = func.copy()

    # segmenting and presenting the found areas
    function_binary[np.where(func > 0)] = 0
    function_binary[np.where(func < 0)] = 1
    function_binary = np.uint8(function_binary)

    contours = measure.find_contours(func, 0.)
    blob_labels = measure.label(function_binary, background = 0)
    label_props = measure.regionprops(blob_labels)

    #     contours = chooseLargestContours(contours, label_props, 1)
    if not hold:
        imshow(img)
    l_curve = []

    for c in contours:
        c[:, [0, 1]] = c[:, [1, 0]]  # swapping between columns: x will be at the first column and y on the second
        if color:
            color = (random.random(), random.random(), random.random())

        curve, = ax.plot(c[:, 0], c[:, 1], '-', color = color)
        l_curve.append(curve)

    return l_curve, ax


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
    return gpd.GeoDataFrame(df, geometry = 'coordinates')



if __name__ == '__main__':
    img_orig = cv2.cvtColor(cv2.imread(r'D:\Documents\ownCloud\Data\Images\Image.bmp'), cv2.COLOR_BGR2GRAY)
    img_normed = cv2.normalize(img_orig.astype('float'), None, 0.0, 1.0,
                               cv2.NORM_MINMAX)  # Convert to normalized floating point

    computeImageDerivatives(img_normed, 1, ksize = 5)
