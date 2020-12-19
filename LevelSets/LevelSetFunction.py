import cv2
import numpy as np

import MyTools as mt

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

    def __init__(self, function, gradientType = 'L2', ksize=3, sigma=2.5, epsilon=1., resolution=1., regularization_note=0, **kwargs):
        r"""

        :param gradientType: the type of gradient computation ('L1', 'L2', 'LoG')
        :param ksize: kernel size for Gaussian blurring
        :param sigma: sigma size for Gaussian blurring
        :param epsilon for regularization threshold, usually the size of the step size
        :param resolution: scanning resolution 
        :param regularization_note: the regularization note (0,1, or 2) for the Heaviside and the Dirac-Delta functions

        :type gradientType: str
        :type ksize: int
        :type sigma: float
        :type epsilon: float
        :type resolution: float
        :type regularization_note: int
        
        :rtype: LevelSetFunction

        """
        self.__gradientType = gradientType
        self.__ksize = ksize
        self.__sigma = sigma
        self.__resolution = resolution
        self.__epsilon = epsilon
        self.__regularization_note = regularization_note
        
        self.__value = function
        self.__ls_derivatives_curvature()
        self.compute_heaviside()
        self.compute_dirac_delta()

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
        gradientType = self.__gradientType
        sigma =  self.__sigma
        ksize = self.__ksize

        if np.all(self.value == 0):
            self.__x = np.zeros(self.value.shape)
            self.__y = np.zeros(self.value.shape)
            self.__xx = np.zeros(self.value.shape)
            self.__yy = np.zeros(self.value.shape)
            self.__xy = np.zeros(self.value.shape)
            self.__kappa = np.zeros(self.value.shape)
            return

        self.__x, self.__y, self.__xx, self.__yy, self.__xy = \
            mt.computeImageDerivatives_numeric(self.value, 2, ksize=ksize,
                                       sigma=sigma, resolution=self.__resolution)

        self.__x = mt.make_zero(self._x)
        self.__y = mt.make_zero(self._y)
        self.__xx = mt.make_zero(self._xx)
        self.__yy = mt.make_zero(self._yy)
        self.__xy = mt.make_zero(self._xy)

        if gradientType == 'L1':
            self.__norm_nabla = cv2.GaussianBlur((np.abs(self.__x) + np.abs(self.__y)), (ksize, ksize), sigma)
            # self.__norm_nabla = (np.abs(self.__x) + np.abs(self.__y))
        elif gradientType == 'L2':
            self.__norm_nabla = cv2.GaussianBlur(np.sqrt(self.__x ** 2 + self.__y ** 2), (ksize, ksize), sigma)
            # self.__norm_nabla = np.sqrt(self.__x ** 2 + self.__y ** 2)
        elif gradientType == 'LoG':
            from scipy.ndimage import filters
            # self.__norm_nabla = cv2.GaussianBlur(filters.gaussian_laplace(self.value, sigma), (ksize, ksize), sigma)
            self.__norm_nabla = filters.gaussian_laplace(self.value, sigma)

        self.__kappa = cv2.GaussianBlur((self._xx * self._y ** 2 + self._yy * self._x ** 2 - 2 * self._xy * self._x * self._y) / (self.norm_nabla + EPS),
                                        (ksize, ksize), sigma)
        # self.__kappa = mt.make_zero(self.kappa, EPS)

        # self.__kappa = (self._xx * self._y ** 2 +
        #                                  self._yy * self._x ** 2 -
        #                                  2 * self._xy * self._x * self._y) / (self.norm_nabla + EPS)

    def update(self, new_function):
        """
        Updates the function according to the new given function

        :param new_function: to update within the self.value
        :param regularization_note: the regularization note (0,1, or 2) for the Heavisdie and the Dirac-Delta functions.

        """
        ksize = self.__ksize
        sigma = self.__sigma
        
        if np.any(new_function.mean()+ 2.5* new_function.std() > new_function.max()):
            new_function[new_function > np.percentile(new_function, 97)] = np.percentile(new_function, 60)
            new_function[new_function< np.percentile(new_function, 3)] = np.percentile(new_function, 40)

        # self.__value = mt.make_zero(cv2.GaussianBlur(new_function,  (ksize, ksize),  sigmaX=sigma))
        self.__value = new_function
        self.__ls_derivatives_curvature()
        
        self.compute_heaviside()
        self.compute_dirac_delta()

    def move_function(self, dphi):
        """
        Adds dphi to the level set function

        :param dphi: the delta to add to the current function

        :type dphi: np.array

        """
        # Phi_temp = LevelSetFunction(self.value + dphi)

        # new_phi = cv2.GaussianBlur(self.value + dphi,
        #                            (self.__ksize, self.__ksize),
        #                            self.__sigma)
        new_phi = self.value + dphi
        self.update(new_phi)

    def reinitialization(self, dphi):
        r"""
        Keeping phi close to a signed function.
        
        :param dphi: the delta to add to the current function

        :type dphi: np.array

        This should keep
        
        .. math::
        
            |\nabla \phi| \approx 1

        Here we use: 
        
        .. math::
            
            \phi_t = S(\phi_0)\left(1- |\nabla \phi| )
        
        where S is a smoothed signed function: 
        
        .. math::
            S(\phi_0) = \frac{\phi_0}{\sqrt{\phi_0 + \epsilon^2}}, \qquad  \epsilon = \min(\Delta x, \Delta y)

        assuming a grid. 
               
        """
        from Utils.MyTools import scale_values

        phi_temp = LevelSetFunction(self.value + dphi)

        # statistical normalization
        # value_temp = phi_temp.value
        # value_temp[phi_temp.value > np.percentile(phi_temp.value, 95)] = np.percentile(phi_temp.value, 90)
        # value_temp[phi_temp.value < np.percentile(phi_temp.value, 5)] = np.percentile(phi_temp.value, 10)
        # phi_temp.update(value_temp)

        S_phi = self.value / np.sqrt(self.value ** 2 + 1)

        phi_t = S_phi * (1- phi_temp.norm_nabla)
        new_value = self.value + phi_t
        # import skfmm
        # new_value = skfmm.distance(self.value + dphi)
        # new_value[new_value<=0] = scale_values(new_value[new_value<=0], -1., -0)
        # new_value[new_value>0] = scale_values(new_value[new_value>0], 0, 1.)
        
        self.update(new_value)
        # while np.any(self.norm_nabla) > 1e10:
        #     new_value = self.value
        #     new_value[self.norm_nabla > 1e10] = np.sign(new_value[self.norm_nabla > 1e10]) * 1
        #     new_value = scale_values(self.value + phi_t, -1.0, 1.0)
        #     self.update(new_value)

    def compute_heaviside(self):
        r"""
        Computes the Heaviside function of phi: H(phi)

        regularization_note:
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



        :param kwargs: epsilon - size to be considered as zero. Default 1e-4

        :return: the heaviside function. Also updates the class.

        """

        epsilon = self.__epsilon
        regularization_note = self.__regularization_note

        H = np.zeros(self.value.shape)
        x = self.value.copy()

        if regularization_note == 0:
            H[x >= 0] = 1

        elif regularization_note == 1:
            H[x > epsilon] = 1
            H[np.abs(x) <= epsilon] = 0.5 * (1 + x[np.abs(x) <= epsilon] / epsilon +
                                      1 / np.pi * np.sin(
                        np.pi * x[np.abs(x) <= epsilon] / epsilon))

        elif regularization_note == 2:
            H = 0.5 * (1 + 2 / np.pi * np.arctan(x / epsilon))

        self.__heaviside = H
        return H

    def compute_dirac_delta(self):
        r"""
        Computes the Dirac Delta function of phi: d(phi)

        regularization_note:
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

        :param epsilon: threshold for a value to be considered zero. Default: 1e-4

        :return: the dirac delta function. Also updates the class

        """
        epsilon = self.__epsilon
        regularization_note = self.__regularization_note

        d = np.zeros(self.value.shape)
        x = self.value.copy()

        if regularization_note == 0:
            d[np.abs(x) <= epsilon] = 1

        elif regularization_note == 1:
            d[np.abs(x) > epsilon] = 0
            d[np.abs(x) <= epsilon] = 1 / (2 * epsilon) * (
                    1 + np.cos(np.pi * x[np.abs(x) <= epsilon] / epsilon))

        elif regularization_note == 2:
            d = 1 / np.pi * epsilon / (epsilon ** 2 + x ** 2)

        elif regularization_note == 3:
            d = self.norm_nabla

        d = mt.make_zero(d, 2e-5)
        self.__dirac_delta = d
        return d

