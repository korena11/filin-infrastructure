"""
Class of different weighting functions for PointNeighborhoods. Returns an object of weighted neighborhood (inherits from neighborhood, only distances are weighted according to the function used here

.. note::
    Can be changed, this was implemented for a neighborhood, but it can be further generalized.
"""
from DataClasses.PointSubSet import PointSubSet
from Properties.Neighborhood.PointNeighborhood import PointNeighborhood

import numpy as np

class WeightedPointNeighborhood(object):
    """
    All weights are centered at the first point of the subset (assuming this is the center of the neighborhood)
    """
    def __init__(self, pointNeighborhood):
        """

        :param pointNeighborhood: the neighborhood for weighting

        :type pointNeighborhood: PointNeighborhood, PointSubSet

        :return: the weights of each point within the neighborhood
        """
        self.weights = None # the weight of each point in the neighborhood
        self.weighted_distances = pointNeighborhood.distances # initializes with uniform distribution
        # weighted distances of the neighborhood
        if isinstance(pointNeighborhood, PointNeighborhood):
            self.pointNeighborhood = pointNeighborhood # the PointNeighborhood to which the function was built
        elif isinstance(pointNeighborhood, PointSubSet):
            self.pointNeighborhood = PointNeighborhood(pointNeighborhood)

    def triangleWeights(self, effectiveDistance):
        """
        A (revolution) triangle function, starting the triangle at the center point, ending at the effective distance (rho). Middle way is highest weight, i.e., 1.

        .. math::
            w(d) = \begin{cases}
            -\frac{2}{\rho}d & d< \frac{\rho}{2} \\
            1 + 2 - \frac{2}{\rho}d & d > \frac{\rho}{2} \\
            1 & d = \frac{\rho}{2}
            \end{cases}

        :param effectiveDistance: size of the triangle's base (the distance in which the points still have a weight larger than zero)

        :type effectiveDistance: float

        :return: weights

        :rtype: float, np.array
        """

        most_effective = effectiveDistance / 2
        dists = self.pointNeighborhood.distances

        weights = np.zeros((self.pointNeighborhood.Size, 1))

        weights[dists<most_effective] = 1/most_effective * dists[dists < most_effective]
        weights[dists == most_effective] = 1
        weights[dists > most_effective] = 3 - 1/most_effective * dists[dists > most_effective]

        self.weights = weights
        self.weighted_distances = weights * dists

        return  weights
