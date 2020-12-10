"""
Different weighting functions for PointNeighborhoods. Returns an object of weighted neighborhood (inherits from neighborhood, only distances are weighted according to the function used here

.. note::
    Can be changed, this was implemented for a neighborhood, but it can be further generalized.
"""
import numpy as np


def triangleWeights(point_neighborhood, effectiveDistance):
    r"""
    A (revolution) triangle function, starting the triangle at the center point, ending at the effective distance (rho). Middle way is highest weight, i.e., 1.

    .. math::
        w(d) = \begin{cases}
        -\frac{2}{\rho}d & d< \frac{\rho}{2} \\
        2 - \frac{2}{\rho}d & d > \frac{\rho}{2} \\
        1 & d = \frac{\rho}{2}
        \end{cases}

    :param effectiveDistance: size of the triangle's base (the distance in which the points still have a weight larger than zero)

    :type effectiveDistance: float

    :return: weights

    :rtype: float, np.array
    """

    most_effective = effectiveDistance / 2
    dists = point_neighborhood.distances

    weights = np.zeros((point_neighborhood.Size, ))

    weights[dists<most_effective] = 1/most_effective * dists[dists < most_effective]
    weights[dists == most_effective] = 1
    weights[dists > most_effective] = 2 - 1/most_effective * dists[dists > most_effective]
    weights[dists > effectiveDistance] = 0

    return weights


def gaussianWeights(pointNeighborhood, sigma, rho):
    r"""
    A weighting function around rho, with width sigma (a Gaussian variant)

    .. math::
        w(x,y) =  \frac{1}{\sqrt{2 \cdot \pi \cdot  \sigma^2}} \cdot  \exp(-0.5 \cdot \frac{\sqrt{x^2+y^2} - rho}{2\sigma^2)}

    :param pointNeighborhood: the neighborhood to weight
    :param sigma: the width of the gaussian
    :param rho: the effective distance

    :type pointNeighborhood: Properties.Neighborhood.PointNeighborhood.PointNeighborhood
    :type sigma: float
    :type rho: float

    :return:
    """

    weights = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(pointNeighborhood.distances - rho)**2 / (2 * sigma ** 2))
    return weights


def laplacianWeights(pointNeighborhood, effectiveDistance):
    r"""
    A weighting function based on the Laplacian with width rho (only the positive parts)


    .. math::
        w(d) =  \frac{1}{\sqrt{2 * np.pi}}cdot \exp{-\frac{d^2}{2}} - \frac{1}{\sqrt{2 \pi \cdot \rho^2}} \cdot \exp{-\frac{d^2}{2 \rho^2}}

    :param pointNeighborhood: the neighborhood to weight
    :param effectiveDistance: the effective distance, after which the weight degrades

    :type pointNeighborhood: Properties.Neighborhood.PointNeighborhood.PointNeighborhood
    :type effectiveDistance: float

    :return: weights
    """

    weights= 1 / np.sqrt(2 * np.pi) * \
           np.exp(-pointNeighborhood.distances ** 2 / 2) - \
           1 / np.sqrt(2 * np.pi * effectiveDistance ** 2) * \
           np.exp(-pointNeighborhood.distances ** 2 / (2 * effectiveDistance ** 2))
    weights[weights<0]=0
    return weights