import numpy as np

import RotationUtils
from Transformations.TransformationMatrixProperty import TransformationMatrixProperty


class TransformationMatrixFactory(object):
    '''
    Different methods for rotation matrix computation
    '''

    @staticmethod
    def Rotation_2Vectors(points, a1, a2):
        '''
        compute rotation matrix from vector a1 to a2
        rotation by angle between a1 and a2 around the unit vector

        :param points: points to rotate
        :param a1: a2: row vectors

        :type points: PointSet

        :return: R rotation matrix

        :rtype: 3x3 nd-array

        '''

        R = RotationUtils.Rotation_2Vectors(a1, a2)
        
        return TransformationMatrixProperty(points, rotationMatrix = R)


    @staticmethod
    def Rotation_EulerAngles(points, angles, dtype='degrees'):
        '''
        Given rotation angles build a rotation matrix

         :param points: points to rotate
         :param angles: omega, phi, kappa
         :param dtype: radians, degrees or symbolic

         .. warning::

             Symbolic option should be checked

         :type points: PointSet
         :type angles: tuple
         :type dtype: str

        :return: R rotation matrix

        :rtype: 3x3 nd-array
        '''
        import RotationUtils
        R = RotationUtils.BuildRotationMatrix(angles[0], angles[1], angles[2])
        return TransformationMatrixProperty(points, rotationMatrix=R)

    @staticmethod
    def Rotation_AxisAngle(points, axis, theta):
        '''
        create rotation matrix given axis and angle

        :param points:  points to rotate
        :param axis: axis to rotate around
        :param theta: rotation angle in degrees

        :type points: PointSet
        :type axis: nx1 or 1xn nd-array
        :type theta: float
            
         :return: R rotation matrix

        :rtype: 3x3 nd-array
        '''
        s = np.sin(np.radians(theta))
        c = np.cos(np.radians(theta))
        t = 1 - c
        x = axis[0]
        y = axis[1]
        z = axis[2]
        R = np.array([[t * x ** 2 + c, t * x * y + s * z, t * x * z - s * y],
                      [t * x * y - s * z, t * y ** 2 + c, t * y * z + s * x],
                      [t * x * z + s * y, t * y * z - s * x, t * z ** 2 + c]])
        return TransformationMatrixProperty(points, rotationMatrix=R)

    @staticmethod
    def Rotation_Quaternion(points, q):
        '''
        create rotation matrix given axis and angle

        :param points: (PointSet) points to rotate
        :param q: quaternion

        :type points: PointSet
        :type q:  np.ndarray 4x1

        :return: R rotation matrix

        :rtype:  np.ndarray 3x3
        '''
        R = np.array([[q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2, 2 * (q[1] * q[2] - q[0] * q[3]),
                       2 * (q[1] * q[3] + q[0] * q[2])],
                      [2 * (q[1] * q[2] + q[0] * q[3]), q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2,
                       2 * (q[3] * q[2] - q[0] * q[1])],
                      [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[3] * q[2] + q[0] * q[1]),
                       q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2]])

        return TransformationMatrixProperty(points, rotationMatrix=R)
