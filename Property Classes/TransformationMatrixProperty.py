import numpy as np

from BaseProperty import BaseProperty


class TransformationMatrixProperty(BaseProperty):
    """
    This class represents a 4x4 3D transformation matrix
    """
    __transformMatrix = np.array(np.zeros((4, 4)))
    __rotationMatrix = None
    __translationMatrix = np.array(np.zeros(1, 3))

    def __init__(self, points, **kwargs):
        super(TransformationMatrixProperty, self).__init__(points)

    def setValues(self, *args, **kwargs):
        """
        Sets the rotation and translation matrices

        :param transformationMatrix: 4x4 transformation matrix

        *Options*

        :param translationMatrix: 1x3 translation matrix
        :param rotationMatrix: 3x3 rotation.

        :type transormationMatrix: np.array
        :type translationMatrix: np.array
        :type rotationMatrix: np.array

        """
        if args:
            self.transformMatrix = args[0]




    @staticmethod
    def FromFile():
        pass
