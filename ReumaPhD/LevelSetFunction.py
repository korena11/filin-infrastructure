import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

import numpy as np
import MyTools as mt
import cv2

EPS = np.finfo(float).eps


class LevelSetFunction(object):
    value = None  # level set function (nd-array)
    _x = _y = _xx = _xy = _yy = None  # level set function derivatives
    norm_nabla = None  # |\nabla \phi| - level set gradient size

    regularization_note = 0  # the regularization note (0,1, or 2) for the Heaviside and the Dirac-Delta functions
    heaviside = None
    dirac_delta = None

    kappa = None  # level set curvature
    processing_props = {}  # processing properties for gradient computation

    def __init__(self, function, regularization_note = 0, **kwargs):
        """
        Builds an initial smooth function (Lipsschitz continuous) that represents the interface as the set where
        phi(x,y,t) = 0 is the curve.
        The function has the following characteristics:
        phi(x,y,t) > 0 for (x,y) \in \Omega
        phi(x,y,t) < 0 for (x,y) \not\in \Omega
        phi(x,y,t) = 0 for (x,y) on curve
        :param regularization_note: the regularization note (0,1, or 2) for the Heaviside and the Dirac-Delta functions

        :param kwargs:
        processing_properties:

        :param sigma
        :param ksize
        :param gradientType
        :param epsilon for regularization threshold

        :return: LevelSetFunction class
        """
        processing_props = {'gradientType': 'L1', 'sigma': 2.5, 'ksize': 5}
        self.processing_props = processing_props
        self.value = function
        self.__ls_derivatives_curvature()
        self.regularization_note = regularization_note
        self.Heaviside(**kwargs)
        self.Dirac_delta(**kwargs)

    def __ls_derivatives_curvature(self, **kwargs):
        """
        computes and updates the level set function derivatives and curvature
        :param processing_props: gradient type, sigma and ksize for derivatives and gradient computations

        """

        self.norm_nabla = mt.computeImageGradient(self.value, gradientType = self.processing_props['gradientType'])
        self._x, self._y, self._xx, self._yy, self._xy = \
            mt.computeImageDerivatives(self.value, 2, ksize = self.processing_props['ksize'],
                                       sigma = self.processing_props['sigma'])

        self.kappa = cv2.GaussianBlur((self._xx * self._y ** 2 +
                                       self._yy * self._x ** 2 -
                                       2 * self._xy * self._x * self._y) /
                                      (self.norm_nabla + EPS),
                                      (self.processing_props['ksize'], self.processing_props['ksize']),
                                      self.processing_props['sigma'])

    def update(self, new_function, **kwargs):
        """
        Updates the function according to the new given function
        :param new_function - to update within the self.value
        :param kwargs:

            :param regularization_note: the regularization note (0,1, or 2) for the Heavisdie and the Dirac-Delta functions.

        """
        self.value = new_function
        self.__ls_derivatives_curvature()
        if 'regularization_note' in kwargs.keys():
            self.regularization_note = kwargs['regularization_note']

        self.Heaviside(**kwargs)
        self.Dirac_delta(**kwargs)

    @staticmethod
    def build_function(func_shape, **kwargs):
        """
        :param func_shape: shape of the function (tuple) (num_row, num_col)
         characteristics of the function:

        :param width: the width of the area inside the curve
        :param height: the height of the area inside the curve
        :param type: 'rectangle' (default); 'vertical' or 'horizontal' (for open contours)

        :return:
        """
        func_type = kwargs.get('func_type', 'rectangle')

        func_width = func_shape[1]
        func_height = func_shape[0]

        width = kwargs.get('width', func_width / 4)
        height = kwargs.get('height', func_height / 4)

        start_point = kwargs.get('start', (func_height / 2 - height, func_width / 2 - width))

        # phi(x,y,t) < 0 for (x,y) \not\in \Omega
        phi = -np.ones(func_shape)

        if func_type == 'rectangle':
            # phi(x,y,t) > 0 for (x,y) \in \Omega
            phi[start_point[0]: start_point[0] + 2 * height, start_point[1]:start_point[1] + 2 * width] = 1

            # phi(x,y,t) = 0 for (x,y) on curve
            phi[start_point[0]: start_point[0] + 2 * height, start_point[1]] = 0
            phi[start_point[0]: start_point[0] + 2 * height, start_point[1] + 2 * width - 1] = 0

            phi[start_point[0], start_point[1]:start_point[1] + 2 * width - 1] = 0
            phi[start_point[0] + 2 * height, start_point[1]:start_point[1] + 2 * width - 1] = 0

        elif func_type == 'horizontal':
            phi[start_point[0], :] = 0
            phi[:start_point[0], :] = 1

        elif func_type == 'vertical':
            phi[:, start_point[1]] = 0
            phi[:, :start_point[1]] = 1

        return phi

    def Heaviside(self, **kwargs):
        """
        Returns the Heaviside function of phi: H(phi)

        :param regularization_note:
         0 - no regularization: H(x) = 1 if >= 0; H(x) = 0 x < 0
         1 - 1st regularization: H(x) = 1 if x > epsilon; H(x)=0 if x< -epsilon;
                                H(x)=0.5*(1+x/epsilon + 1/pi*sin(pi*x / epsilon)) if |x| <= epsilon
         2 - 2nd regularization: H(x) = 0.5*(1+2/pi * arctan(x/epsilon))
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

        self.heaviside = H
        return H

    def Dirac_delta(self, **kwargs):
        """
        Returns the Dirac Delta function of phi. d(phi)

        :param regularization_note:
         0 - no regularization: d(x) = phi if  x >=0; d(x) = 0 if x < 0
         1 - 1st regularization: d(x) = 0 if  |x| > epsilon;
                                d(x) = 1/(2*epsilon) *(1+cos(pi*x / epsilon)) if |x|<=epsilon
         2 - 2nd regularization: d(x) = 1/pi * epsilon/(epsilon**2 + x**2)
        :param kwargs: epsilon - for the 1st and 2nd regularizations
        :return: the dirac delta function. Also updates the class
        """
        epsilon = kwargs.get('epsilon', EPS)
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

        self.dirac_delta = d
        return d
