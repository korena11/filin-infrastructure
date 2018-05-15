from warnings import warn

import numpy as np

from BaseProperty import BaseProperty

class RotationMatrixProperty( BaseProperty ):
    """
    .. warning:: Now only implemented for Euler angles.
    """

    __rotationType = 'euler'  # The type of rotation matrix (euler, quaternion, axis and rotation, etc.)

    # TODO: add angles extraction in case of other rotations than "euler"

    def __init__(self, points, rotation_matrix, rotation_axis = None):
        super(RotationMatrixProperty, self).__init__(points)

        self.setValues(rotation_matrix, rotation_axis)

    def setValues(self, *args, **kwargs):
        """
        Sets the rotation matrix for the RotationMatrixProperty object

        :param rotationMatrix: a 3x3 matrix that holds the rotation
        :param rotationAxis: a 3x1 vector that holds the axis about which the rotation is made, in a quaternion case

        *Optionals*

        :param rotation_type: 'quaternions', 'euler'

        :type matrix_type: str
        :type rotationMatrix: np.array
        :type rotationAxis: np.array

        """
        self.__rotation_matrix = args[0]
        rotationAxis = args[1]

        if rotationAxis is None:
            warn('No rotation type was set for the matrix. Default is Euler Rotation Matrix')

        else:
            if 'rotation_type' in kwargs:
                self.__rotationType = kwargs['rotation_type']

            else:
                warn('No rotation type was set for the matrix, but the matrix has'
                     'axis of rotation Default is Euler Rotation Matrix')

    @property
    def RotationMatrix( self ):
        """
        Points' rotation matrix with respect to reference pointset
        """
        print self.__rotationType
        return self.__rotation_matrix

    def EulerAngles_from_R(self, dtype = 'degrees'):
        """
        Extracts rotation angles from the rotation matrix

        .. warning:: The angles returned are subjected to ambiguity

        :param dtype: the output in 'degrees' or 'radians' (default 'degrees')

        :type dtype: str

        :return:  (omega, phi, kappa) according to the dtype

        :rtype: tuple

        """

        warn('The angles may contain ambiguity')
        if self.__rotationType == 'euler':
            R = self.__rotation_matrix
        else:
            pass
            # TODO: add the quaternion or other rotation options

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
