"""
This module contains derivatives computations on curves and images
"""

import numpy as np
from numpy.linalg import norm
from scipy.ndimage import filters

import cv2
from Utils.MyTools import repartitionCurve


def curveCentralDerivatives(c, repartition=5):
    '''
    Computes the first and second central derivative of a curve
    :param c: curve points, not necessarily arc-length
    :param repartition: the step for repartition. if negative does not change the curve partition
    :return: tuple, (c', c")
    '''
    dc = np.diff(c, axis=0)
    dtau = norm(dc, axis=1)
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


def imageDerivatives_Sobel(img, order, **kwargs):
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
              'window': (0, 0)}
    params.update(kwargs)
    ksize = np.int(params['ksize']) * params['resolution']

    img = np.float64(img)

    img_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    img_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

    if params['sigma'] != 0:
        img_x = cv2.GaussianBlur(img_x, params['window'], params['sigma'])
        img_y = cv2.GaussianBlur(img_y, params['window'], params['sigma'])

    if order == 2:
        img_xx = cv2.Sobel(img_x, cv2.CV_64F, 1, 0, ksize=ksize)
        img_yy = cv2.Sobel(img_y, cv2.CV_64F, 0, 1, ksize=ksize)
        img_xy = cv2.Sobel(img_x, cv2.CV_64F, 0, 1, ksize=ksize)

        if params['sigma'] != 0:
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


def imageGradient_numeric(img, derivativesFunc, gradientType='L1', ksize=3, sigma=1, resolution=1, blur_window=(0, 0), **kwargs):
    """
        :param img: the image to which the derivatives should be computed
        :param derivativesFunc: the function with which the derivatives are computed.
        :param gradientType: 'L1' L1 norm of grad(I); 'L2' L2-norm of grad(I); 'LoG' Laplacian of gaussian
        :param ksize: size of the differentiation window
        :param resolution: kernel resolution
        :param sigma: sigma for gaussian blurring. Default: 1. If sigma=0 no smoothing is carried out
        :param blur_window: tuple of window size for blurring

        :type img: np.array
        :type derivativesFunc: function
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
    dx, dy = derivativesFunc(img, 1, ksize, sigma, resolution, blur_window)

    if gradientType == 'L1':
        gradient = cv2.GaussianBlur((np.abs(dx) + np.abs(dy)), (ksize, ksize), sigma)  # L1-norm of grad(I)
    elif gradientType == 'L2':
        gradient = cv2.GaussianBlur(np.sqrt(dx ** 2 + dy ** 2), (ksize, ksize), sigma)
    elif gradientType == 'LoG':
        gradient = filters.gaussian_laplace(img, sigma)

    # return cv2.normalize((gradient).astype('float'), None, 0.0,1.0, cv2.NORM_MINMAX)
    return gradient


def imageDerivatives_4connected(img, order, ksize=3, sigma=1., resolution=1., blur_window=(0, 0), **kwargs):
    """
        Computes numeric image derivatives up to order 2, with 4 neighbors.

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
    I1 = np.hstack((img[:, floord:], img[:, - floord:]))
    I2 = np.hstack((img[:, : floord], img[:, : - floord]))
    Zx = (I1 - I2) / (2 * resolution * np.ceil(ksize))

    # y direction
    I3 = np.vstack((img[floord:, :], img[-floord:, :]))
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


def imageSecondDerivatives_8connected(img, ksize=3, sigma=None, resolution=1., blur_window=(0, 0), **kwargs):
    """
        Computes numeric image derivatives up to order 2, with 8 neighbors.

        Larger windows will have 8 neighbors at the edges of the window

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
    """
    img = np.float64(img)
    ksize = np.int(ksize * resolution)

    if ksize % 2 ==0: # make sure that the window is odd number
        ksize +=1

    # build the window
    window = np.zeros((ksize,ksize))
    center_pix = (ksize-1)/2
    # first column
    window[0,0] = -1
    window[0, center_pix] = -1
    window[0,-1] = -1

    # middle column
    window[center_pix, 0] = -1
    window[center_pix, 0, center_pix] = 8
    window[center_pix, -1] = -1

    # last column
    window[-1, 0] = -1
    window[-1, center_pix] = -1
    window[-1, -1] = -1

    # convolve the window with the image
    


