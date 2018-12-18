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

        # --------- To make the object iterable ---------
        self.current = 0
        # --------- To make the object iterable ---------
        self.current = 0

    # ---------- Definitions to make iterable -----------
    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        try:
            return self.getTensor(self.current - 1)
        except IndexError:
            self.current = 0
            raise StopIteration

    # --------end definitions for iterable object-----------

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

    def GetAllPointsTensors(self):
        """
        Retrieve all points tensors

        :rtype: np.array
        """
        return self.__tensors
