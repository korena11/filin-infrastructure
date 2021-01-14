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
    sin_xx = np.sin(np.pi / 25 * xx)
    sin_yy = np.sin(np.pi / 25 * yy)

    return sin_xx * sin_yy


def dist_from_circles(dx, dy, radius, func_shape, resolution=.5):
    r"""
    Build a Lipshitz distance function with repetitive circles

    .. math::

       \phi(x,y,t) < 0 \quad \text{for } (x,y) \not\in \Omega

    :param dx: distance between circles on x
    :param dy: distance between circles on y
    :param radius: radius of the circles
    :param func_shape: size of the function (height, width)
    :param resolution: grid size

    :return: a level set function that its zero-set is the defined circles (approximately)
    """

    import skfmm
    from tqdm import tqdm
    phi = -np.ones(func_shape)
    center_x = np.arange(dx/2 + radius, func_shape[1] , dx + radius)
    center_y = np.arange(dy/2 + radius, func_shape[0] , dy + radius)

    for i in tqdm(center_y, position=0, leave=False):
        for j in tqdm(center_x, position=1, leave=True):
            phi_temp = dist_from_circle((i,j), radius, func_shape, resolution=resolution)
            phi[int(i-radius):int(i+radius),int(j-radius):int(j+radius)] = phi_temp[int(i-radius):int(i+radius),int(j-radius):int(j+radius)]

    return skfmm.distance(phi)