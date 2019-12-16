import warnings

import numpy as np
import sympy
from sympy.matrices import Matrix


def decdeg2dms(dd):
    """
    Convert from decimal degrees to degree-minute-second
    :param dd: degree in decimal degrees
    :type dd: float

    :return: degree in dms

    :rtype: tuple (degree, minutes, seconds)
    """

    is_positive = dd >= 0
    dd = abs(dd)
    minutes, seconds = divmod(dd * 3600, 60)
    degrees, minutes = divmod(minutes, 60)
    degrees = degrees if is_positive else -degrees
    return (degrees, minutes, seconds)


# TODO: sit with Elia and compare to answer in the url. Something is not right
def Rotation_2Vectors(from_vector, to_vector):
    r'''
    Computes rotation matrix from 'fromVector' to 'toVector'

    Rotation by angle between 'fromVector' and 'toVector' around the unit vector
    - `Link to information
    <https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d>_`

    .. math::

        \bf{R} = \bf{I} + \sin(\alpha) \cdot [v]_\times + (1-\cos(\alpha))\cdot[v]_\times^2

    with :math:`[v]_\times` skewsymmetric inner product matrix
    (i.e., :math:`v=a_\text{from vector}\cdot b_\text{to vector}`; and :math:`\alpha` the
    angle between from and to vectors

    :param from_vector: the vector from which the rotation matrix should be computed
    :param to_vector: the vector to which the matrix should rotate to

    :type from_vector: np.ndarray
    :type to_vector: np.ndarray

    :return: rotation matrix
    :rtype: np.ndarray
    '''

    # Normalize both vectors
    from_vector = from_vector / np.linalg.norm(from_vector)
    to_vector = to_vector / np.linalg.norm(to_vector)

    # Cross Product - Results in a unit vector
    cross_product_unit_vector = np.cross(from_vector, to_vector)  # = Sin(AngleBetweenFromToVectors)
    cos_of_angle = np.sum(
        from_vector * to_vector)  # = Cos(AngleBetweenFromToVectors) # Slightly faster than np.dot in short vectors

    if cos_of_angle == 1:
        # if np.array_equal(from_vector, to_vector):
        return np.eye(3)
    elif cos_of_angle == -1:
        return -np.eye(3)

    h = (1.0 - cos_of_angle) / (1.0 - cos_of_angle ** 2)  # If scalarProduct is zero, this will crash

    R = np.array([[cos_of_angle + h * cross_product_unit_vector[0] ** 2,
                   h * cross_product_unit_vector[0] * cross_product_unit_vector[1] - cross_product_unit_vector[2],
                   h * cross_product_unit_vector[0] * cross_product_unit_vector[2] + cross_product_unit_vector[1]],
                  [h * cross_product_unit_vector[0] * cross_product_unit_vector[1] + cross_product_unit_vector[2],
                   cos_of_angle + h * cross_product_unit_vector[1] ** 2,
                   h * cross_product_unit_vector[1] * cross_product_unit_vector[2] - cross_product_unit_vector[0]],
                  [h * cross_product_unit_vector[0] * cross_product_unit_vector[2] - cross_product_unit_vector[1],
                   h * cross_product_unit_vector[1] * cross_product_unit_vector[2] + cross_product_unit_vector[0],
                   cos_of_angle + h * cross_product_unit_vector[2] ** 2]])

    return R


def Skew(v):
    """
    Builds a skew symmetric matrix from an array

    :param v: 3x1 array
    :type v: nd-array

    :return skew symmetric matrix

    """

    skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
    return skv - skv.T


def RotationByOmega(omega, dtype='degrees'):
    """
    Builds a rotation matrix about the x-axis

    :param omega: angle
    :param dtype: angles units ('radians' / 'degrees' / 'symbolic'). Default: 'degrees'

    :type omega: float
    :type dtype: str

    :return: a 3x3 rotation matrix, around the x-axis

    :rtype: np.array
    """

    if dtype == 'symbolic':
        # sympy.init_session(quiet=True)
        # rotation around x
        R_omega = Matrix([[1, 0, 0],
                          [0, sympy.cos(omega), -sympy.sin(omega)],
                          [0, sympy.sin(omega), sympy.cos(omega)]])

    else:

        if dtype == 'degrees':
            omega = np.radians(omega)

        # rotation around x
        R_omega = np.array([[1, 0, 0],
                            [0, np.cos(omega), -np.sin(omega)],
                            [0, np.sin(omega), np.cos(omega)]])
    return R_omega


def RotationByPhi(phi, dtype='degrees'):
    """
    Builds a rotation matrix about the y-axis

    :param phi: angle
    :param dtype: angles units ('radians' / 'degrees' / 'symbolic'). Default: 'degrees'

    :type phi: float
    :type dtype: str

    :return: a 3x3 rotation matrix, around the y-axis
    """

    if dtype == 'symbolic':

        # rotation around y
        R_phi = Matrix([[sympy.cos(phi), 0, sympy.sin(phi)],
                        [0, 1, 0],
                        [-sympy.sin(phi), 0, sympy.cos(phi)]])

    else:
        if dtype == 'degrees':
            phi = np.radians(phi)

        # rotation around y
        R_phi = np.array([[np.cos(phi), 0, np.sin(phi)],
                          [0, 1, 0],
                          [-np.sin(phi), 0, np.cos(phi)]])

    return R_phi


def RotationByKappa(kappa, dtype='degrees'):
    """
    Builds a rotation matrix about the z-axis

    :param kappa: angle
    :param dtype: angles units ('radians' / 'degrees' / 'symbolic'). Default: 'degrees'

    :type kappa: float
    :type dtype: str

    :return: a 3x3 rotation matrix, around the z-axis
    """
    if dtype == 'symbolic':
        # sympy.init_session(quiet=True)

        # rotation around z
        R_kappa = Matrix([[sympy.cos(kappa), -sympy.sin(kappa), 0],
                          [sympy.sin(kappa), sympy.cos(kappa), 0],
                          [0, 0, 1]])
    else:

        if dtype == 'degrees':
            kappa = np.radians(kappa)

        R_kappa = np.array([[np.cos(kappa), -np.sin(kappa), 0],
                            [np.sin(kappa), np.cos(kappa), 0],
                            [0, 0, 1]])
    return R_kappa


def BuildRotationMatrix(omega, phi, kappa, dtype='degrees'):
    r"""
    Builds rotation matrix from image to world, according to a z-y'-x" rotation:

    .. math:: {\bf R}_{\omega, \phi, \kappa} = \begin{bmatrix}
                1 & 0 & 0 \\
                0 & \cos(\omega) & -\sin(\omega) \\
                0 & \sin(\omega) & \cos(\omega)
                \end{bmatrix}  \begin{bmatrix}
                \cos(\phi) & 0 & \sin(\phi) \\
                0 & 1 & 0\\
                \sin(\phi) & 0 & \cos(\phi)
                \end{bmatrix} \begin{bmatrix}
                 \cos(\kappa) & -\sin(\kappa) & 0\\
                 -\sin(\kappa) & \cos(\kappa) & 0\\
                    0 & 0 & 1
                \end{bmatrix}

    :param omega: rotation about x-axis
    :param phi: rotation about y-axis
    :param kappa: rotation about z-axis
    :param dtype: angles units ('radians' / 'degrees' / 'symbolic'). Default: 'degrees'

    :type omega: float
    :type phi: float
    :type kappa: float
    :type dtype: str

    :return: rotation matrix 3x3 nd-array
    """

    R_omega = RotationByOmega(omega, dtype)
    R_phi = RotationByPhi(phi, dtype)
    R_kappa = RotationByKappa(kappa, dtype)

    return R_omega.dot(R_phi.dot(R_kappa))


def ExtractRotationAngles(R, dtype='degrees'):
    """
    Extracts rotation angles from the rotation matrix

    .. note:: The angles returned are subjected to ambiguity

    :param R: rotation matrix
    :param dtype: should the output be in 'degrees' or 'radians' (default 'degrees')

    :type R: nd-array 3x3
    :type dtype: str

    :return:  (omega, phi, kappa) according to the dtype
    """

    warnings.warn('The angles may contain ambiguity')
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
