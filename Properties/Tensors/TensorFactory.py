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

from PointSet import PointSet
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

        ** Optionals **

        :param radius: the radius according to which weights are being computed (if not set, then unit weight is used)

        :type points: PointSet
        :type point_index: int

        :return: TensorProperty object

        :rtype: Tensor
        """

        if point_index == -1:
            ref_point = np.mean(points.ToNumpy(), axis=0)  # Computing the center of gravity of the points

        else:
            ref_point = points.GetPoint(point_index)

        deltas = points.ToNumpy() - ref_point

        # Set weights, if needed:
        if 'radius' in kwargs:
            sigma = kwargs['radius'] / 3
            w = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
                -(deltas[:, 0] ** 2 + deltas[:, 1] ** 2 + deltas[:, 2] ** 2) / (2 * sigma ** 2))
            if np.sum(np.isnan(w)) > 0 or np.sum(np.isinf(w)) > 0 or np.abs(np.sum(w)) < 1e-10:
                w = np.ones(points[:, 0].shape)
        else:
            w = np.ones(points.Size)

        # Compute the covariance matrix of points around the ref_point
        covMat = (w[:, None] * deltas).T.dot(deltas) / np.sum(w)

        t = Tensor(covMat, ref_point, points.Size)

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
    def computeTensorsProperty_givenNeighborhood(points, neighborhoodProperty, **kwargs):
        """
        Compute tensors for a point cloud.

        For each point a tensor is computed, with the point itself the reference point.

        :param points: a point cloud (PointSet or sub-object)
        :param neighborhoodProperty: a property that holds all neighbors for each point

        :type points: PointSet
        :type neighborhoodProperty:
        :type knn: int

        :return: a tensor property with all tensors computed for each point


        .. code-block:: python

            pts = np.random.rand(1000, 3) - 0.5) * 1000.0
            points = PointSet(pts)



        """

        tensors = TensorProperty(points)

        for i in np.arange(points.Size):
            pass
