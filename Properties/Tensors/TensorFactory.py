'''
infragit
reuma\Reuma
|today|

.. note::
     Base on Zachi's implementation

The factory either creates a TensorProperty or a tensor for each point in a point cloud.  

'''

import platform

import numpy as np

from NeighborProperty import NeighborsProperty
from PointSet import PointSet
from PointSetOpen3D import PointSetOpen3D
from TensorProperty import TensorProperty
from Tensors.tensor import Tensor

if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')


class TensorFactory(object):
    """

    """

    @classmethod
    def tensorFromPoints(cls, points, point_index=-1, **kwargs):
        """
        Create a Tensor from 3D points

        :param points: PointSet
        :param point_index: around which point index the tensor is computed.  If (-1) the tensor is computed around the
           center of gravity. Default (-1).

        **Optionals**

        :param radius: the radius according to which weights are being computed (if not set, then unit weight is used)

        :type points: PointSet
        :type point_index: int

        :return: TensorProperty object

        :rtype: Tensor

                """

        if point_index == -1:
            ref_point = -1
            points_array = points.ToNumpy()

        else:
            ref_point = points.GetPoint(point_index)
            points_array = points.ToNumpy()
            points_array1 = points_array[point_index + 1:, :]
            points_array2 = points_array[:point_index, :]
            points_array = np.vstack((points_array1, points_array2))

        # points_array = points.ToNumpy()
        deltas = points_array - ref_point

        # Set weights, if needed:
        if 'radius' in kwargs:
            sigma = kwargs['radius'] / 3
            w = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
                -(deltas[:, 0] ** 2 + deltas[:, 1] ** 2 + deltas[:, 2] ** 2) / (2 * sigma ** 2))
            if np.sum(np.isnan(w)) > 0 or np.sum(np.isinf(w)) > 0 or np.abs(np.sum(w)) < 1e-10:
                w = np.ones(points_array[:, 0].shape)
        else:
            w = 1

        t = TensorFactory.tensorGeneral(points_array, ref_point, weights=w)

        return t

    @classmethod
    def tensorGeneral(cls, arrays, ref_array=-1, weights=1):
        """
        Compute a general tensor (not necessarily 3D)

        :param arrays: an nxm arrays (each *row* array is the set for the tensor)
        :param ref_array: the array around which the tensor is computed, if (-1) it is computed around the center of gravity

        :type arrays: np.array
        :type ref_array: np.array

        **Optionals**
        :param weights: the weights for the tensor computation. Default 1 for all

        :return: a tensor

        :rtype: Tensor
        """

        if isinstance(ref_array, int) and ref_array == -1:
            ref_array = np.mean(arrays, axis=1)

        deltas = arrays - ref_array

        # Set weights, if needed:
        if isinstance(weights, int) and weights == 1:
            weights = np.ones(arrays.shape[0])

        # Compute the covariance matrix of the arrays around the ref_array
        covMat = (weights[:, None] * deltas).T.dot(deltas) / np.sum(weights)

        t = Tensor(covMat, ref_array, arrays.shape[0])

        return t


    @classmethod
    def joinTensors(cls, t1, t2):
        """
        Create a new Tensor by joining two existing ones

        :param t1: First TensorSegment object
        :param t2: Second TensorSegment object

        :type t1: Tensor
        :type t2: Tensor

        :return: A new Tensor object which is the result of merging the two given ones

        :rtype: Tensor


        """
        if not isinstance(t1, Tensor) and isinstance(t2, Tensor):
            raise TypeError('Argument must be a TensorSegment object')

        n1 = t1.points_number
        n2 = t2.points_number

        covMat = n1 * t1.covariance_matrix / (n1 + n2) + n2 * t2.covariance_matrix / (n1 + n2) + \
                 n1 * n2 ** 2 * np.dot((t2.reference_point - t1.reference_point).reshape((-1, 1)),
                                       (t2.reference_point - t1.reference_point).reshape((1, -1))) / \
                 (n1 + n2) ** 3 + \
                 n1 ** 2 * n2 * np.dot((t2.reference_point - t1.reference_point).reshape((-1, 1)),
                                       (t2.reference_point - t1.reference_point).reshape((1, -1))) / \
                 (n1 + n2) ** 3

        return Tensor(covMat, (n1 * t1.reference_point + n2 * t2.reference_point) / (n1 + n2), n1 + n2)

    @staticmethod
    def computeTensorsProperty_givenNeighborhood(points, neighborhoodProperty=None, neighborsFunc=None, **kwargs):
        """
        Compute tensors for a point cloud.

        For each point a tensor is computed, with the point itself the reference point. If the neighborhoodProperty is
        None, it is computed here, according to the sent function.

        :param points: a point cloud (PointSet or sub-object)

        **Optionals**

        :param neighborhoodProperty: a property that holds all neighbors for each point (can be empty)
        :param neighborsFunc: the function to be used for neighbors search.
        :param kwargs: the arguments for the function to be used for neighbors search. These are usually:
            :param radius: search radius for neighbors search
            :param knn: k nearest neighbors for neighbors search

        :type points: PointSet
        :type neighborhoodProperty: NeighborhoodProperty
        :type knn: int
        :type radius: float
        :type neighborsFunc: function

        :return: a tensor property with all tensors computed for each point

        :rtype: TensorProperty

        **Usage**

        .. warning::

            Now works only for PointSet3D neighborhood search.

        """

        tensors = TensorProperty(points)

        if isinstance(neighborhoodProperty, NeighborsProperty):
            # in case the neighborhood was already defined
            for i in np.arange(points.Size):
                neighbors = neighborhoodProperty.getNeighbors(i)
                if neighbors == None:
                    continue
                radius = neighbors.radius
                tensors.setValues(i, TensorFactory.tensorFromPoints(neighbors.neighbors, i, radius=radius))

        else:
            # compute neighbors and tensor at the same time
            for i in np.arange(points.Size):
                neighborhoodProperty = NeighborsProperty(points)

                if not isinstance(points, PointSetOpen3D):
                    points = PointSetOpen3D(points)

                neighbors = neighborsFunc(points, i, *kwargs, neighborsProperty=neighborhoodProperty)
                radius = neighbors.radius
                tensors.setValues(i, TensorFactory.tensorFromPoints(neighbors.neighbors, i, radius=radius))
        return tensors
