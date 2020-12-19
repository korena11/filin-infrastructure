"""
A module for the creation of distance function
"""

import numpy as np


def dist_from_circle(center_pt, radius, func_shape, resolution=.5):
    r"""
    Build a Lipshitz distance function from a circle, with a specific size

    .. math::

       \phi(x,y,t) < 0 \quad \text{for } (x,y) \not\in \Omega


    :param center_pt:  center of the circle
    :param radius:  radius of the circle
    :param func_shape: size of the function (height, width)
    :param resolution: the kernel size for later processing. Default: 0.5

    :type center_pt: tuple
    :type radius: int
    :type func_shape: tuple

    :return: a level set function that its zero-set is the defined circle (approximately)

    :rtype: np.array

    """

    height = func_shape[0]
    width = func_shape[1]
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(resolution * x, resolution* y)
    x_x0 = (xx - center_pt[1] * resolution) # (x-x0)
    y_y0 =  (yy - center_pt[0] * resolution)  # (y-y0)

    phi = radius *resolution - np.sqrt(x_x0 ** 2 + y_y0 ** 2)

    return phi

def dist_from_ellipse(center_pt, axes, func_shape, resolution=.5):
    r"""
    Build a Lipshitz distance function from an ellipse, with a specific size

    .. math::

       \phi(x,y,t) < 0 \quad \text{for } (x,y) \not\in \Omega


    :param center_pt:  center of the ellipse
    :param axes:  axes sizes of the ellipse
    :param func_shape: size of the function (height, width)
    :param resolution: the kernel size for later processing. Default: 5

    :type center_pt: tuple
    :type radius: int
    :type func_shape: tuple

    :return: a level set function that its zero-set is the defined ellipse (approximately)

    :rtype: np.array

    """
    height = func_shape[0]
    width = func_shape[1]
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(resolution * x, resolution * y)
    x_x0 = (xx - center_pt[1] * resolution)  # (x-x0)
    y_y0 = (yy - center_pt[0] * resolution)  # (y-y0)

    phi = np.sqrt((x_x0 / axes[0]) ** 2 + (y_y0 / axes[1]) ** 2) - 1

    return phi

def dist_from_checkerboard(func_shape):
    r"""
    Build a Lipshitz distance function with a sine function, with a specific size

    .. math::

       \phi({\bf x}) = \sin\left(\frac{\pi}{5}x\right)\cdot \sin\left(\frac{\pi}{5}y\right)


    :param amplitude:  the amplitude of the egg grate
    :param func_shape: size of the function (height, width)
    :param resolution: the kernel size for later processing. Default: .5

    :type center_pt: tuple
    :type radius: int
    :type func_shape: tuple

    :return: a level set function that its zero-set is the defined circle (approximately)

    :rtype: np.array
    """
    height = func_shape[0]
    width = func_shape[1]
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    sin_xx = np.sin(np.pi / 5 * xx)
    sin_yy = np.sin(np.pi / 5 * yy)

    return sin_xx * sin_yy