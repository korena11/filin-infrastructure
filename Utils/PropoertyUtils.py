"""
Utilities for handling properties
"""

from DataClasses.KdTreePointSet import KdTreePointSet, PointSet
from Properties.BaseProperty import BaseProperty
from Properties.Neighborhood.NeighborsFactory import NeighborsFactory, PointNeighborhood, NeighborsProperty

from tqdm import tqdm, trange
import numpy as np


def smoothProperty(property, smoothing_operator, neighborhood_size=None, kmax=None, attributes=None, **kwargs):
    """
    Smooths properties' values in all attributes, unless one or more attributes are specified.

    Examples for properties that can be smoothed: ColorProperty, SaliencyProperty, CurvatureProperty.

    :param property: the property to smooth
    :param smoothing_operator: what kind of smoothing should take place (Gaussian, median, mean, etc.) _others can be added_.
    :param neighborhood_size: the neighborhood over which the smoothing will carry (equivalent to window size)
    :param kmax: maximal number of neighbors in the neighborhood. According to :func:`NeighborsFactory.kdtreePointSet_rnn`
    :param attributes: one or more specific attributes to smooth. If not specified -- all attributes will be smoothed.
    :param kwargs: if needed, more arguments for other smoothing functions that require more parameters.
        - :param neighborhood: if the neighborhood was computed in advance
        - :param parts_size: the size of the parts to compute neighbors (see :func:`NeiborsFactory.kdtreePointSet_rnn`)
        - :param parts_num: the number of parts to computed neighbors (see :func:`NeiborsFactory.kdtreePointSet_rnn`)

    :type property: BaseProperty
    :type smoothing_operator: func
    :type neighborhood_size: float
    :type kmax: int
    :type attributes: str, list
    :type neighborhood: NeighborsProperty
    :type parts_size: int
    :type parts_num: int


    :return: Property with smoothed values
    :rtype: property.__class__

    """
    import inspect
    # default values for neighborhood computation
    parts_size = int(5e5)
    parts_num = None

    kdpts = KdTreePointSet(property.Points)

    if 'neighborhood' in kwargs.keys():
        neighborhoodProperty = kwargs['neighborhood']
    else:
        if 'parts_size' in kwargs.keys():
            parts_size = kwargs['parts_size']
        elif 'parts_num' in kwargs.keys():
            parts_num = kwargs['parts_num']

        neighborhoodProperty = NeighborsFactory.kdtreePointSet_rnn(kdpts, neighborhood_size, kmax, parts_size, parts_num)

    # create the new smoothed property
    smoothed_property = property.__class__(property.Points)

    if attributes is None:
        attributes_inclass = inspect.getmembers(property.__class__, lambda a:(inspect.isdatadescriptor(a))) # all members of the class (not including methods)
        attributes = [a for a in attributes_inclass if not hasattr(BaseProperty, a[0])] # only attributes that are not in the BaseProperty

    new_attributes = {}

    for attribute in attributes:
        newvals = []
        values = np.asarray(property.__getattribute__(attribute[0]))

        # in each neighborhood, smooth according to the smoothing operator
        for neighborhood in tqdm(neighborhoodProperty, 'smoothing' + attribute[0], neighborhoodProperty.Size):

            idx = neighborhood.neighborhoodIndices
            newvals.append(smoothing_operator(values[idx], **kwargs))

        newvals = np.asarray(newvals)
        new_attributes.update({attribute[0]: newvals})
    smoothed_property.load(**new_attributes)
    return smoothed_property


def median_smoothing_property(values, **kwargs):
    """
    Median smoothing for a neighborhood.

    :param values: the values to smooth

    :type values: np.array, list

    :return: the median value of ``values``

    :rtype: float

    """
    return np.median(values)


def mean_smoothing_property(values, **kwargs):
    """
    Apply mean smoothing on a neighborhood

    :param values: the values to smooth

    :type values: np.array, list

    :return: the mean value of ``values``
    :rtype: float

    """
    return np.mean(values)

# TODO: def gaussian_smoothing_property(values, neighborhood, sigma):
