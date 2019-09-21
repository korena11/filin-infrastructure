"""
|today|

.. codeauthor:: Reuma

.. note::
     Based on Zachi's implementation


A property for a whole point cloud. Does not have to have all tensors computed for all points. However, if a tensor is
not computed, a PointTensor should be constructed and inserted here for the respective point
"""

import numpy as np

from BaseProperty import BaseProperty
from Tensor import Tensor


class TensorProperty(BaseProperty):

    def __init__(self, points, tensors=None, **kwargs):
        super(TensorProperty, self).__init__(points)

        self.__tensors = np.empty(points.Size, Tensor)
        if tensors is not None:
            self.load(tensors)

    def __next__(self):
        self.current += 1
        try:
            return self.getPointTensor(self.current - 1)
        except IndexError:
            self.current = 0
            raise StopIteration

    def load(self, tensors, **kwargs):
        """
        Sets an entire tensor array to the object

        :param tensors: an entire array to set

        :type tensors: np.ndarray of Tensor.Tensor

        """
        self.__tensors = np.asarray(tensors)

    def setPointTensor(self, idx, values):
        """
        Sets the tensor for an index

        :param idx: the index to which the tensor should be updated
        :param values: the computed tensor

        :type idx: int
        :type values: Tensor

        """

        self.__tensors[idx] = values

    def getPointTensor(self, idx):
        """
        Retrieve a tensor of point(s) with idx index

        :param idx: the index of the point

        :type idx: int

        :return: the tensor of point idx

        :rtype: Tensor
        """
        return self.__tensors[idx]

    def GetAllPointsTensors(self):
        """
        Retrieve all points tensors

        :rtype: np.array
        """
        return self.__tensors
