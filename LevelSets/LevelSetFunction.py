import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

import numpy as np
import MyTools as mt
import cv2

EPS = np.finfo(float).eps


class LevelSetFunction(object):
    r"""
    Builds an initial smooth function (Lipsschitz continuous) that represents the interface as the set where

    .. math::
        \phi(x,y,t) = 0

    is the curve.

    The function has the following characteristics:

    .. math::
       \begin{cases}
        \phi(x,y,t) > 0 & \textrm{for} & (x,y) \in \Omega\\
        \phi(x,y,t) < 0 & \textrm{for} & (x,y) \not\in \Omega\\
        \phi(x,y,t) = 0 & \textrm{for} & (x,y) \textrm{ on curve}
        \end{cases}

    """

    def __init__(self, function, regularization_note = 0, **kwargs):
        r"""


        :param regularization_note: the regularization note (0,1, or 2) for the Heaviside and the Dirac-Delta functions

        :param processing_properties:

        :param sigma:
        :param ksize:
        :param gradientType:
        :param epsilon for regularization threshold:

        :rtype: LevelSetFunction

        """
        processing_props = {'gradientType': 'L1', 'sigma': 2.5, 'ksize': 1}
        self.__processing_props = processing_props
        self.__value = function
        self.__ls_derivatives_curvature()
        self.__regularization_note = regularization_note
        self.compute_heaviside(**kwargs)
        self.compute_dirac_delta(**kwargs)

    @property
    def value(self):
        """
        Function's value (nd-array)

        """
        return self.__value

    @property
    def norm_nabla(self):
        r"""
        Function's gradient size

        .. math::
           |\nabla \phi|

        """
        return self.__norm_nabla

    @property
    def kappa(self):
        """
        Function's curvature

        """
        return self.__kappa

    @property
    def processing_props(self):
        """
        Processing properties for gradient computation

        :rtype: dict

        """
        return self.__processing_props

    @property
    def dirac_delta(self):
        """
        Function's dirac-delta according to the regularization note

        """
        return self.__dirac_delta

    @property
    def heaviside(self):
        """
        Function's heaviside according to the regularization note

        """
        return self.__heaviside

    @property
    def regularization_note(self):
        """
        Regularization note (0,1, or 2) for the Heaviside and the Dirac-Delta functions
        """
        return self.__regularization_note

    @property
    def _x(self):
        """
        Function's differentiation by x

        """
        return self.__x

    @property
    def _y(self):
        """
        Function's differentiation by y

        """
        return self.__y

    @property
    def _xx(self):
        """
        Function's 2nd differentiation by xx

        """
        return self.__xx

    @property
    def _xy(self):
        """
        Function's 2nd differentiation by xy

        """
        return self.__xy

    @property
    def _yy(self):
        """
        Function's 2nd differentiation by xx

        """
        return self.__yy

    def __ls_derivatives_curvature(self, **kwargs):
        """
        Computes and updates the level set function derivatives and curvature

        :param processing_props: gradient type, sigma and ksize for derivatives and gradient computations

        """

        self.__norm_nabla = mt.computeImageGradient(self.value, gradientType = self.processing_props['gradientType'])
        self.__x, self.__y, self.__xx, self.__yy, self.__xy = \
            mt.computeImageDerivatives(self.value, 2, ksize = self.processing_props['ksize'],
                                       sigma = self.processing_props['sigma'])

        self.__kappa = cv2.GaussianBlur((self._xx * self._y ** 2 +
                                         self._yy * self._x ** 2 -
                                         2 * self._xy * self._x * self._y) /
                                        (self.norm_nabla + EPS),
                                        (self.processing_props['ksize'], self.processing_props['ksize']),
                                        self.processing_props['sigma'])

    def update(self, new_function, **kwargs):
        """
        Updates the function according to the new given function

        :param new_function: to update within the self.value
        :param regularization_note: the regularization note (0,1, or 2) for the Heavisdie and the Dirac-Delta functions.

        """
        self.__value = new_function
        self.__ls_derivatives_curvature()
        if 'regularization_note' in list(kwargs.keys()):
            self.__regularization_note = kwargs['regularization_note']

        self.compute_heaviside(**kwargs)
        self.compute_dirac_delta(**kwargs)

    def move_function(self, dphi):
        """
        Adds dphi to the level set function

        :param dphi: the delta to add to the current function

        :type dphi: np.array

        """
        new_phi = cv2.GaussianBlur(self.value + dphi, (self.processing_props['ksize'], self.processing_props['ksize']),
                                   self.processing_props['sigma'])
        self.update(new_phi)

    @staticmethod
    def dist_from_circle(center_pt, radius, func_shape, **kwargs):
        """
        Build a Lipshitz distance function from a circle, with a specific size

        .. math::
             \phi(x,y,t) < 0 \quad \text{for } (x,y) \not\in \Omega


        :param center_pt:  center of the circle
        :param radius:  radius of the circle
        :param func_shape: size of the function (height, width)
        :param ksize: the kernel size for later processing. Default: 5

        :type center_pt: tuple
        :type radius: int
        :type func_shape: tuple

        :return: a level set function that its zero-set is the defined circle (approximately)

        :rtype: np.array

        """
        height = func_shape[0]
        width = func_shape[1]
        ksize = kwargs.get('ksize', 5)
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(ksize * x, ksize * y)

        phi = radius - np.sqrt((xx - center_pt[1] * ksize) ** 2 + (yy - center_pt[0] * ksize) ** 2)

        return phi

    def compute_heaviside(self, **kwargs):
        r"""
        Computes the Heaviside function of phi: H(phi)

        :param regularization_note:
            - '0' - no regularization:
                .. math::
                  \begin{cases}
                  H(x) = 1 & \textrm{if} &x >= 0\\
                  H(x) = 0 & \textrm{if}& x < 0
                  \end{cases}

            - '1' - 1st regularization:
               .. math::
                  \begin{cases}
                  H(x) = 1 & \textrm{if} & x > \epsilon\\
                  H(x)=0 & \textrm{if} & x< -\epsilon\\
                  H(x)=0.5\cdot\left[1+\frac{x}{\epsilon} + \frac{1}{\pi}\cdot\sin\left(\frac{\pi\cdot x}{\epsilon}\right)\right] & \textrm{if} & |x| <= \epsilon
                  \end{cases}

            - '2' - 2nd regularization:
                .. math::
                    H(x) = 0.5\cdot\left[1+\frac{2}{\pi} \cdot \arctan\left(\frac{x}{\epsilon}\right)\right]



        :param kwargs: epsilon - for the regularizations

        :return: the heaviside function. Also updates the class.

        """
        epsilon = kwargs.get('epsilon')
        regularization_note = kwargs.get('regularization_note', self.regularization_note)

        H = np.zeros(self.value.shape)
        x = self.value.copy()

        if regularization_note == 0:
            H[x >= 0] = 1

        elif regularization_note == 1:
            H[x > epsilon] = 1
            H[x >= -epsilon] = 0.5 * (1 + x[x >= -epsilon] / epsilon +
                                      1 / np.pi * np.sin(
                        np.pi * x[x >= -epsilon] / epsilon))
            H[x <= epsilon] = 0.5 * (1 + x[x <= epsilon] / epsilon +
                                     1 / np.pi * np.sin(
                        np.pi * x[x <= epsilon] / epsilon))

        elif regularization_note == 2:
            H = 0.5 * (1 + 2 / np.pi * np.arctan(x / epsilon))

        self.__heaviside = H
        return H

    def compute_dirac_delta(self, **kwargs):
        r"""
        Computes the Dirac Delta function of phi: d(phi)

        :param regularization_note:
            - '0' - no regularization:
                .. math::
                   \begin{cases}
                   d(x) = \phi & \textrm{if} &  x >=0\\
                   d(x) = 0 & \textrm{if} & x < 0
                   \end{cases}

            - '1' - 1st regularization:
                .. math::
                   \begin{cases}
                   d(x) = 0 & \textrm{if} &  |x| > \epsilon\\
                   d(x) = \frac{1}{2\epsilon}\cdot\left[1+\cos\left(\frac{\pi\cdot x}{\epsilon}\right)\right] & \textrm{if} & |x|<=\epsilon
                   \end{cases}

            - '2' - 2nd regularization:
                .. math::
                   d(x) = \frac{1}{\pi}\frac{\epsilon}{\epsilon^2 + x^2}

            - '3' - replace dirac-delta with :math:`|\nabla \phi|`

        :param epsilon: for the 1st and 2nd regularizations

        :return: the dirac delta function. Also updates the class

        """
        epsilon = kwargs.get('epsilon', 1)
        regularization_note = kwargs.get('regularization_note', self.regularization_note)

        d = np.zeros(self.value.shape)
        x = self.value.copy()

        if regularization_note == 0:
            d[x >= 0] = x[x >= 0]

        elif regularization_note == 1:
            d[x >= -epsilon] = 1 / (2 * epsilon) * (
                    1 + np.cos(np.pi * x[x >= -epsilon] / epsilon))
            d[x <= epsilon] = 1 / (2 * epsilon) * (
                    1 + np.cos(np.pi * x[x <= epsilon] / epsilon))

        elif regularization_note == 2:
            d = 1 / np.pi * epsilon / (epsilon ** 2 + x ** 2)

        elif regularization_note == 3:
            d = self.norm_nabla

        self.__dirac_delta = d
        return d
