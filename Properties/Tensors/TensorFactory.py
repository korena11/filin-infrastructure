'''
infragit
reuma\Reuma
|today|

.. note::
     Base on Zachi's implementation

The factory either creates a TensorProperty or a tensor for each point in a point cloud.  

'''

import numpy as np

from Properties.Neighborhood.NeighborsProperty import NeighborsProperty
from DataClasses.PointSet import PointSet
from Properties.Tensors.TensorProperty import TensorProperty
from Properties.Tensors.Tensor import Tensor


class TensorFactory(object):
    """

    """

    @classmethod
    def tensorFromPoints(cls, points, point_index=-1, keepPoints=True, **kwargs):
        """
        Create a Tensor from 3D points

        :param points: PointSet
        :param point_index: around which point index the tensor is computed.  If (-1) the tensor is computed around the
           center of gravity. Default (-1).

        **Optionals**

        :param radius: the radius according to which weights are being computed (if not set, then unit weight is used)
        :param keepPoints: indicator whether to keep the points as part of the tensor object

        :type points: PointSet or np.ndarray
        :type point_index: int
        :type keepPoints: bool

        :return: Tensor object

        :rtype: Tensor

        """

        if point_index == -1:
            if isinstance(points, PointSet):
                points_array = points.ToNumpy()
            elif isinstance(points, np.ndarray):
                points_array = points
            else:
                raise TypeError('Unexpected type of \'points\' object')
            ref_point = -1

            if points_array.shape[0] == 0:
                raise ValueError('Number of points should be at least one')

        else:
            ref_point = points.GetPoint(point_index)
            points_array = points.ToNumpy()

            # TODO: check validity and remove redundant code
            local_idx = np.nonzero(ref_point in points_array)
            points_array1 = points_array[local_idx[0][0] + 1:, :]
            points_array2 = points_array[:local_idx[0][0], :]
            points_array = np.vstack((points_array1, points_array2))

        # Set weights, if needed:
        if 'radius' in kwargs:
            # points_array = points.ToNumpy()
            deltas = points_array - ref_point

            sigma = kwargs['radius'] / 3
            w = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
                -(deltas[:, 0] ** 2 + deltas[:, 1] ** 2 + deltas[:, 2] ** 2) / (2 * sigma ** 2))

            if np.sum(np.isnan(w)) > 0 or np.sum(np.isinf(w)) > 0 or np.abs(np.sum(w)) < 1e-10:
                w = np.ones(points_array[:, 0].shape)
        else:
            w = 1

        t, _ = TensorFactory.tensorGeneral(points_array, ref_point, weights=w, keepPoints=keepPoints)

        return t

    @classmethod
    def tensorGeneral(cls, arrays, ref_array=-1, weights=1, min_points=3, keepPoints=True):
        """
        Compute a general tensor (not necessarily 3D)

        :param arrays: an nxm arrays (each *row* array is the set for the tensor)
        :param ref_array: the array around which the tensor is computed, if (-1) it is computed around the center of gravity

        :type arrays: np.array
        :type ref_array: np.array

        **Optionals**

        :param weights: the weights for the tensor computation. Default 1 for all
        :param min_points: minimal number of points that can define a tensor (default: 3)
        :param keepPoints: indicator whether to keep the points as part of the tensor object

        :type min_points: int
        :type keepPoints: bool

        :return: a tensor

        :rtype: Tensor

        .. note::
            If there are less than `min_point` the covariance matrix of the tensor will be defined as the Identity matrix (with the required shape)
        """
        weights_axis = False  # flag for weights axes size

        if isinstance(ref_array, int) and ref_array == -1:
            if len(arrays.shape) >= 2:
                ref_array = np.mean(arrays, axis=0)
            else:
                ref_array = np.mean(arrays, axis=1)

        deltas = arrays - ref_array

        # Set weights, if needed:
        if isinstance(weights, int) and weights == 1:
            weights = np.ones(arrays.shape[0])

        while weights_axis is False:
            weights = weights[:, np.newaxis]
            if len(weights.shape) == len(deltas.shape):
                weights_axis = True
        # Compute the covariance matrix of the arrays around the ref_array
        covMat = (weights * deltas).T.dot(deltas) / np.sum(weights)

        if arrays.shape[0] < min_points:
            covMat = np.eye(covMat.shape[0])

        if keepPoints:
            t = Tensor(covMat, ref_array, arrays.shape[0], points=arrays)
        else:
            t = Tensor(covMat, ref_array, arrays.shape[0])

        return t, ref_array


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
            raise TypeError('Argument must be a Tensor object')

        n1 = np.int64(t1.points_number)
        n2 = np.int64(t2.points_number)

        covMat = n1 * t1.covariance_matrix / (n1 + n2) + n2 * t2.covariance_matrix / (n1 + n2) + \
                 n1 * n2 ** 2 * np.dot((t2.reference_point - t1.reference_point).reshape((-1, 1)),
                                       (t2.reference_point - t1.reference_point).reshape((1, -1))) / \
                 (n1 + n2) ** 3 + \
                 n1 ** 2 * n2 * np.dot((t2.reference_point - t1.reference_point).reshape((-1, 1)),
                                       (t2.reference_point - t1.reference_point).reshape((1, -1))) / \
                 (n1 + n2) ** 3

        return Tensor(covMat, (n1 * t1.reference_point + n2 * t2.reference_point) / (n1 + n2), n1 + n2)


    @classmethod
    def unifyTensors(cls, tensorList):
        """
        Create a new tensor by joining multiple ones
        :param tensorList: List of Tensor objects (list/ndarray)
        :return: A new tensor which is the result of merging the given list of tensors
        """
        if not np.all(list(map(lambda t: isinstance(t, Tensor), tensorList))):
            raise TypeError('All arguments in list must be a Tensor object')

        # getting the number of points that each tensor consist
        nPnts = np.array(list(map(lambda t: t.points_number, tensorList)))
        totalNumPnts = nPnts.sum()  # computing the total number of points

        # getting the reference points and the covariance matrices of each tensor
        refPnts = np.array(list(map(lambda t: t.reference_point, tensorList)))
        covMats = np.array(list(map(lambda t: t.covariance_matrix, tensorList)))

        # computing the reference point of the unified tensor, set is the weighted average of the individual ones
        unifiedRefPnt = np.dot(nPnts.reshape((1, -1)), refPnts).reshape((-1, )) / totalNumPnts

        # computing the covariance matrix of the unified tensor
        deltas = refPnts - unifiedRefPnt
        translateTensors = np.array(list(map(lambda d: np.dot(d.reshape((-1, 1)), d.reshape((1, -1))), deltas)))
        unifiedCovMat = np.array(
            list(map(lambda t, dt, n: n * t + n * dt, covMats, translateTensors, nPnts))).sum(axis=0) / totalNumPnts

        return Tensor(unifiedCovMat, unifiedRefPnt, totalNumPnts)

    @staticmethod
    def computeTensorsProperty_givenNeighborhood(points, neighborhoodProperty=None, neighborsFunc=None,
                                                 equalWeights=True, **kwargs):
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
        from PointSetOpen3D import PointSetOpen3D
        print(""">>> Computing tensors for all points""")

        tensors = TensorProperty(points)

        if isinstance(neighborhoodProperty, NeighborsProperty):
            # in case the neighborhood was already defined
            for i in np.arange(points.Size):

                neighbors = neighborhoodProperty.getNeighborhood(i)

                if neighbors is None:
                    continue
                if equalWeights:
                    tensors.setPointTensor(i, TensorFactory.tensorFromPoints(neighbors.neighbors, 0))
                else:
                    radius = neighbors.radius
                    tensors.setPointTensor(i, TensorFactory.tensorFromPoints(neighbors.neighbors, 0, radius=radius))

        else:
            # compute neighbors and tensor at the same time
            for i in np.arange(points.Size):
                neighborhoodProperty = NeighborsProperty(points)

                if not isinstance(points, PointSetOpen3D):
                    points = PointSetOpen3D(points)

                neighbors = neighborsFunc(points, i, *kwargs, neighborsProperty=neighborhoodProperty)

                if equalWeights:
                    tensors.load(i, TensorFactory.tensorFromPoints(neighbors.neighbors, i))
                else:
                    radius = neighbors.radius
                    # always around the first points, as the point around which the tensor is looked is the first on the
                    # neighbors list (the closest to itself)
                    tensors.load(i, TensorFactory.tensorFromPoints(neighbors.neighbors, i, radius=radius))
        return tensors
