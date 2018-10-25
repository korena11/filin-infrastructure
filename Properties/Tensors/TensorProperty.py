"""
|today|

.. codeauthor:: Reuma

.. note::
     Base on Zachi's implementation


A property for a whole point cloud. Does not have to have all tensors computed for all points. However, if a tensor is
not computed, a PointTensor should be constructed and inserted here for the respective point
"""

import numpy as np

from BaseProperty import BaseProperty
from tensor import Tensor


class TensorProperty(BaseProperty):

    def __init__(self, points, tensors=None):
        super(TensorProperty, self).__init__(points)

        self.__tensors = np.empty(points.Size, Tensor)

    def setValues(self, idx, tensor):
        """
        Sets the tensor for an index

        :param idx: the index to which the tensor should be updated
        :param tensor: the computed tensor

        :type idx: int
        :type tensor: Tensor

        """

        self.__tensors[idx] = tensor

    def getTensor(self, idx):
        """
        Retrieve a tensor of point(s) with idx index

        :param idx: the index of the point

        :type idx: int

        :return: the tensor of point idx

        :rtype: Tensor
        """
        return self.__tensors[idx]
