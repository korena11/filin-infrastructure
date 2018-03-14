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


def imshow(img, scale_val = 1, ax = None, cmap = 'gray', *args, **kwargs):
    '''
    shows pixel value in the image window
    '''
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(img, cmap = cmap, *args, **kwargs)
    ax.format_coord = lambda x, y: 'x=%d,y=%d, z=%f' % (scale_val * int(x + .5),
                                                        scale_val * int(y + .5), img[y, x])
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


def linearImageCorrection(img, contrast = 1, brightness = 0):
    '''

    @param img: the image for contrast manipulation
    @param contrast: the contrast parameter
    @param brightness: the brightness parameter

    @return: image after contrast manipulation
    '''

    correctedImage = img * contrast + brightness

    if img.dtype == 'float':
        return im2double(correctedImage)
    else:
        return correctedImage.astype(img.dtype)


def imageCorrection(img, phi = 1, theta = 1, contrastType = 'decrease'):
    '''

    @param img: the image for contrast manipulation
    @param contrast:  decrease intensity - dark pixels become much darker and bright pixels become slightly dark OR
                      increase intensity - dark pixels become much brighter, bright pixels become slightly bright
    @param phi, theta: parameters for manipulating data

    @return: image after contrast manipulation
    '''

    if img.dtype == 'uint8':
        maxIntensity = 255.0
    else:
        maxIntensity = 1.0

    # TODO other types

    x = np.linspace(0, maxIntensity)

    if contrastType == 'decrease':
        correctedImage = (maxIntensity / phi) * (img / (maxIntensity / theta)) ** 2
    elif contrastType == 'increase':
        correctedImage = (maxIntensity / phi) * (img / (maxIntensity / theta)) ** 0.5

    if img.dtype == 'float':
        return im2double(correctedImage)
    else:
        return correctedImage.astype(img.dtype)


def computeImageGradient(I, **kwargs):
    '''
    Computes the gradient to a given image

    :param I: image to which the gradient is computed
    :param ksize: kernel size, for blurring and derivatives
    :param sigma: sigma for LoG gradient
    :param gradientType: 'L1' L1 norm of grad(I); 'L2' L2-norm of grad(I); 'LoG' Laplacian of gaussian
    :return: an image of the gradient magnitude
    '''

    ksize = kwargs.get('ksize', 5)
    gradientType = kwargs.get('gradientType', 'L1')
    sigma = kwargs.get('sigma', 2.5)
    gradient = None

    img = cv2.GaussianBlur(I, (0, 0), sigma)

    # compute image gradient
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = ksize)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = ksize)

    if gradientType == 'L1':
        gradient = cv2.GaussianBlur((np.abs(dx) + np.abs(dy)), (0, 0), sigma)  # L1-norm of grad(I)
    elif gradientType == 'L2':
        gradient = cv2.GaussianBlur(np.sqrt(dx ** 2 + dy ** 2), (0, 0), sigma)
    elif gradientType == 'LoG':
        gradient = cv2.GaussianBlur(filters.gaussian_laplace(I, sigma), (0, 0), sigma)

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
        Computes image derivatives up to order 2
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



def is_pos_semidef(x):
    """
    check if a matrix is positive semidefinite
    :param x: the matrix
    :return:  Boolean: True if it is, False is not.
    """
    return np.all(np.linalg.eigvals(x) >= 0)


def hillshade(array, **kwargs):
    """

    :param array: the raster
    :param kwargs:
    :return:
    """
    azimuth = kwargs.get('azimuth', 315.0)
    angle_altitude = kwargs.get('altitude', 45.)

    azimuth = 360.0 - azimuth

    x, y = np.gradient(array)
    slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180.
    altituderad = angle_altitude * np.pi / 180.

    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope) * np.cos(
        (azimuthrad - np.pi / 2.) - aspect)

    return 255 * (shaded + 1) / 2


if __name__ == '__main__':
    img_orig = cv2.cvtColor(cv2.imread(r'D:\Documents\ownCloud\Data\Images\Image.bmp'), cv2.COLOR_BGR2GRAY)
    img_normed = cv2.normalize(img_orig.astype('float'), None, 0.0, 1.0,
                               cv2.NORM_MINMAX)  # Convert to normalized floating point

    computeImageDerivatives(img_normed, 1, ksize = 5)
