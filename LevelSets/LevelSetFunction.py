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

    def __init__(self, function, regularization_note=0, **kwargs):
        r"""


        :param regularization_note: the regularization note (0,1, or 2) for the Heaviside and the Dirac-Delta functions

        :param processing_properties:

        :param sigma:
        :param ksize:
        :param gradientType:
        :param epsilon for regularization threshold:

        :rtype: LevelSetFunction

        """
        processing_props = {'gradientType': 'L1', 'sigma': 2.5, 'ksize': 1, 'resolution': 0.5}
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
        :param epsilon: when kappa is considered zero. Default 1e-4
        """
        gradientType = self.processing_props['gradientType']
        sigma =  self.processing_props['sigma']
        ksize = self.processing_props['ksize']
        epsilon = kwargs.get('eps', 1e-4)

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
                                       sigma=sigma, resolution=self.__processing_props['resolution'])

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

        self.__kappa = cv2.GaussianBlur((self._xx * self._y ** 2 +
                                         self._yy * self._x ** 2 -
                                         2 * self._xy * self._x * self._y) /
                                        (self.norm_nabla + EPS),
                                        (self.processing_props['ksize'], self.processing_props['ksize']),
                                        self.processing_props['sigma'])
        self.__kappa = mt.make_zero(self.kappa, epsilon)

        # self.__kappa = (self._xx * self._y ** 2 +
        #                                  self._yy * self._x ** 2 -
        #                                  2 * self._xy * self._x * self._y) / (self.norm_nabla + EPS)

    def update(self, new_function, **kwargs):
        """
        Updates the function according to the new given function

        :param new_function: to update within the self.value
        :param regularization_note: the regularization note (0,1, or 2) for the Heavisdie and the Dirac-Delta functions.

        """
        if np.any(new_function.mean()+ 2.5* new_function.std() > new_function.max()):
            new_function[new_function > np.percentile(new_function, 97)] = np.percentile(new_function, 60)
            new_function[new_function< np.percentile(new_function, 3)] = np.percentile(new_function, 40)

        self.__value = mt.make_zero(cv2.GaussianBlur(new_function,  (self.processing_props['ksize'], self.processing_props['ksize'])
                                        , sigmaX=self.processing_props['sigma']))
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
        # Phi_temp = LevelSetFunction(self.value + dphi)

        new_phi = cv2.GaussianBlur(self.value + dphi,
                                   (self.processing_props['ksize'], self.processing_props['ksize']),
                                   self.processing_props['sigma'])
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
        from Utils.MyTools import scale_values, make_zero

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
        new_value[new_value<=0] = scale_values(new_value[new_value<=0], -1., -0)
        new_value[new_value>0] = scale_values(new_value[new_value>0], 0, 1.)
        # new_value = make_zero(new_value)


        self.update(new_value)
        # while np.any(self.norm_nabla) > 1e10:
        #     new_value = self.value
        #     new_value[self.norm_nabla > 1e10] = np.sign(new_value[self.norm_nabla > 1e10]) * 1
        #     new_value = scale_values(self.value + phi_t, -1.0, 1.0)
        #     self.update(new_value)


    @staticmethod
    def dist_from_circle(center_pt,  radius, func_shape, resolution=.5):
        r"""
        Build a Lipshitz distance function from a circle, with a specific size

        .. math::
        
           \phi(x,y,t) < 0 \quad \text{for } (x,y) \not\in \Omega


        :param center_pt:  center of the circle
        :param radius:  radius of the circle
        :param func_shape: size of the function (height, width)
        :param resolution: the kernel size for later processing. Default: 0.5

        :type center_pt: tuple
        :type radius: int
        :type func_shape: tuple

        :return: a level set function that its zero-set is the defined circle (approximately)

        :rtype: np.array

        """

        height = func_shape[0]
        width = func_shape[1]
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(resolution * x, resolution* y)
        x_x0 = (xx - center_pt[1] * resolution) # (x-x0)
        y_y0 =  (yy - center_pt[0] * resolution) #(y-y0)

        phi = radius *resolution - np.sqrt(x_x0 ** 2 + y_y0 ** 2)

        return phi

    @staticmethod
    def dist_from_circles(dx, dy, radius, func_shape, resolution=.5):
        r"""
        Build a Lipshitz distance function with repetitive circles

        .. math::

           \phi(x,y,t) < 0 \quad \text{for } (x,y) \not\in \Omega

        :param dx: distance between circles on x
        :param dy: distance between circles on y
        :param radius: radius of the circles
        :param func_shape: size of the function (height, width)
        :param resolution: grid size

        :return: a level set function that its zero-set is the defined circles (approximately)
        """

        import skfmm
        from tqdm import tqdm
        phi = -np.ones(func_shape)
        center_x = np.arange(dx/2 + radius, func_shape[1] , dx + radius)
        center_y = np.arange(dy/2 + radius, func_shape[0] , dy + radius)

        for i in tqdm(center_y, position=0, leave=False):
            for j in tqdm(center_x, position=1, leave=True):
                phi_temp = LevelSetFunction.dist_from_circle((i,j), radius, func_shape, resolution=resolution)
                phi[int(i-radius):int(i+radius),int(j-radius):int(j+radius)] = phi_temp[int(i-radius):int(i+radius),int(j-radius):int(j+radius)]

        return skfmm.distance(phi)

    @staticmethod
    def dist_from_ellipse(center_pt, axes, func_shape, resolution=.5):
        r"""
        Build a Lipshitz distance function from an ellipse, with a specific size

        .. math::

           \phi(x,y,t) < 0 \quad \text{for } (x,y) \not\in \Omega


        :param center_pt:  center of the ellipse
        :param axes:  axes sizes of the ellipse
        :param func_shape: size of the function (height, width)
        :param resolution: the kernel size for later processing. Default: 5

        :type center_pt: tuple
        :type radius: int
        :type func_shape: tuple

        :return: a level set function that its zero-set is the defined ellipse (approximately)

        :rtype: np.array

        """
        height = func_shape[0]
        width = func_shape[1]
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(resolution * x, resolution * y)
        x_x0 = (xx - center_pt[1] * resolution)  # (x-x0)
        y_y0 = (yy - center_pt[0] * resolution)  # (y-y0)

        phi = np.sqrt((x_x0 / axes[0])** 2 + (y_y0/axes[1]) ** 2) - 1

        return phi

    @staticmethod
    def dist_from_eggcrate(amplitude, func_shape, resolution=.5):
        r"""
        Build a Lipshitz distance function with a sine function, with a specific size

        .. math::

           \phi(x,y,t) < 0 \quad \text{for } (x,y) \not\in \Omega


        :param amplitude:  the amplitude of the egg grate
        :param func_shape: size of the function (height, width)
        :param resolution: the kernel size for later processing. Default: .5

        :type center_pt: tuple
        :type radius: int
        :type func_shape: tuple

        :return: a level set function that its zero-set is the defined circle (approximately)

        :rtype: np.array
        """
        height = func_shape[0]
        width = func_shape[1]
        x = np.arange(-width/2, width/2, resolution)
        y = np.arange(-height/2, height/2, resolution)
        xx, yy = np.meshgrid(x, y)
        sin_xx = np.sin(xx )
        sin_yy = np.sin(yy )

        return xx**2 + yy**2 - amplitude * (sin_xx**2 + sin_yy**2)

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



        :param kwargs: epsilon - size to be considered as zero. Default 1e-4

        :return: the heaviside function. Also updates the class.

        """

        epsilon = kwargs.get('epsilon', 1e-04)
        regularization_note = kwargs.get('regularization_note', self.regularization_note)

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

        :param epsilon: threshold for a value to be considered zero. Default: 1e-4

        :return: the dirac delta function. Also updates the class

        """
        epsilon = kwargs.get('epsilon', 1e-04)
        regularization_note = kwargs.get('regularization_note', self.regularization_note)

        d = np.zeros(self.value.shape)
        x = self.value.copy()

        if regularization_note == 0:
            d[np.abs(x) <= epsilon] = 1

        elif regularization_note == 1:
            d[x==0] = 1
            d[np.abs(x)<epsilon] = 0
            d[np.abs(x) > epsilon] = 1 / (2 * epsilon) * (
                    1 + np.cos(np.pi * x[np.abs(x) > epsilon] / epsilon))

        elif regularization_note == 2:
            d = 1 / np.pi * epsilon / (epsilon ** 2 + x ** 2)

        elif regularization_note == 3:
            d = self.norm_nabla

        self.__dirac_delta = d
        return d

