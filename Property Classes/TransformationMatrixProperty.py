import numpy as np

from BaseProperty import BaseProperty


class TransformationMatrixProperty(BaseProperty):
    """
    This class represents a 4x4 3D transformation matrix
    """

    def __init__(self, points, e00, e01, e02, e10, e11, e12, e20, e21, e22):
        super(TransformationMatrixProperty, self).__init__(points)
        transformMatrix = np.matrix(np.zeros((3, 4)))

    @staticmethod
    def FromFile():
        pass
