from warnings import warn

import numpy as np

from BaseProperty import BaseProperty


class TransformationMatrixProperty(BaseProperty):
    """
    This class represents a 4x4 3D transformation matrix
    """
    __transformMatrix = np.array(np.zeros((4, 4)))

    def __init__(self, points, transformationMatrix=None, translationMatrix=None, rotationMatrix=None):
        """
        
        :param points: the point set 
        :param transformationMatrix: 4x4 transformation matrix which includes both rotation and translation 
        :param translationMatrix: 3x1 or 3x3 translation matrix 
        :param rotationMatrix:  3x3 rotation matrix
        
        :type points: PointSet
        :type transformationMatrix: np.ndarray
        :type translationMatrix: np.ndarray
        :type rotationMatrix: np.ndarray
        
        """
        super(TransformationMatrixProperty, self).__init__(points)
        if transformationMatrix is not None:
            self.load(transformationMatrix)

        if rotationMatrix is not None:
            self.load(rotationMatrix)

        if translationMatrix is not None:
            self.load(translationMatrix)

    def load(self, *args, **kwargs):
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
            self.__transformMatrix = args[0]

        if 'translationMatrix' in kwargs:
            shape = kwargs['translationMatrix']
            self.__transformMatrix[-1, :shape[0]] = kwargs['translationMatrix']

        if 'rotationMatrix' in kwargs:
            shape = kwargs['rotationMatrix']
            self.__transformMatrix[:3, :shape[1]] = kwargs['rotationMatrix']

    @property
    def transformationMatrix(self):
        return self.__transformMatrix

    @property
    def translationMatrix(self, dimensions = 3):
        """
        Points' translation matrix with respect to reference pointset

        :param dimensions: the dimensions of the returned vector. Either 3 or 4

        :type dimensions: int

        """
        return self.__transformMatrix[-1, :dimensions]

    @property
    def rotationMatrix(self, dimensions = 3):
        """
        Points' rotation matrix with respect to reference pointset

        :param dimensions: the dimensions of the returned vector. Either 3 or 4

        :type dimensions: int

        """
        return self.__transformMatrix[:3, :dimensions]

    def eulerAngles_from_R(self, dtype = 'degrees'):
        """
        Extracts euler rotation angles from the rotation matrix

        .. warning:: The angles returned are subjected to ambiguity

        :param dtype: the output in 'degrees' or 'radians' (default 'degrees')

        :type dtype: str

        :return:  (omega, phi, kappa) according to the dtype

        :rtype: tuple

        """

        warn('The angles may contain ambiguity')
        R = self.__transformMatrix[:3, :3]
        phi = np.arcsin(R[0, 2])

        if phi >= np.pi:
            n = 1
            phi = np.pi - phi
        else:
            n = 0

        omega = np.arctan(-R[1, 2] / R[2, 2]) + np.pi * n
        kappa = np.arctan2(-R[0, 1], R[0, 0]) + np.pi * n

        if dtype == 'degrees':
            return np.degrees(omega), np.degrees(phi), np.degrees(kappa)

        else:
            return omega, phi, kappa

    @staticmethod
    def FromFile():
        pass
