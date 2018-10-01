'''
infragit
reuma\Reuma (according to Zachi's code)
10, Jul, 2018 
'''

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

import numpy as np
from functools import partial
import numpy.linalg as la
from TensorProperty import TensorProperty
from PointSet import PointSet


class TensorFactory:
    """
    .. warning::

       THIS FACTORY IS NOT FINISHED AND CANNOT BE USED.
    """

    @classmethod
    def tensorFromPoints(cls, points, point_index=-1, **kwargs):
        """
        Creating a tensor instance from a list of 3D points

        :param points: PointSet
        :param point_index: around which point index the tensor is computed.  If (-1) the tensor is computed around the
           center of gravity. Default (-1).

        ** Optionals **

        :param radius: the radius according to which weights are being computes (if not set, then unit weight is used)

        :type points: PointSet
        :type point_index: int

        :return: TensorProperty object

        :rtype: TensorProperty
        """
        numPnts = points.Size[0]
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

        t = TensorProperty(points, covMat, ref_point)

        # Compute the stick radius of the tensor
        norms = np.array(list(map(la.norm, deltas)))
        distances = norms * np.sin(
            np.array(list(map(np.arccos, map(partial(np.dot, b=t.stick_axis), deltas) / norms))))
        t.__stickRadius = np.mean(distances)

        # Computing the sphere parameters defined by the tensor
        t.__sphereRadius = np.mean(map(la.norm, deltas))

        rotMat = np.array([t.eigenvectors[:, 1].reshape((-1,)),
                           t.eigenvectors[:, 2].reshape((-1,)),
                           t.eigenvectors[:, 0].reshape((-1,))])

        temp = np.array(map(rotMat.dot, deltas)).T
        t.__length, t.__width, t.__height = temp.max(1) - temp.min(1)

        # if t.__variance > 0.05 and t.__variance < 0.1 and t.__type == 'plate':
        #     VisualizeTensor(points, self)

        return t
