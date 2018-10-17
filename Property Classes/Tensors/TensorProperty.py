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

    def __init__(self, points):
        super(TensorProperty, self).__init__(points)

        self.__tensors = np.empty(points.Size, Tensor)
