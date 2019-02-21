import numpy as np

from BaseProperty import BaseProperty


class CurvatureProperty(BaseProperty):
    '''
    Curvature property initially holds only the principal curvatures. *However* if an additional type of curvature
    is transferred in "kwargs" it will be assigned, with its name (as sent by kwargs) and an additional dictionary.

    '''
    __principal_curvatures = None
    __invalid_value = -999
    __normalize = False

    def __init__(self, points, principal_curvatures=None, **kwargs):
        super(CurvatureProperty, self).__init__(points)

        self.__principal_curvatures = np.empty((self.Size, 2))
        self.load(principal_curvatures, **kwargs)
        self.__invalid_value = -999  # value for not-computed curvature, default -999
        self.__normalize = False  # flag whether to normalize the principal curvatures

    def __next__(self):
        self.current += 1
        try:
            return self.getPointCurvature(self.current - 1)
        except IndexError:
            self.current = 0
            raise StopIteration

    def set_invalid_value(self, value):
        """
        Value for curvature that was not computed

        :param value: a default value for not compute curvature

        :return:
        """
        self.__invalid_value = value

    def load(self, principal_curvatures, **kwargs):
        """
        Sets curvature into Curvature Property object

        If a new attribute is sent within kwargs, it will be set as a new attribute to the property

        """
        if principal_curvatures is not None:
            self.__principal_curvatures = principal_curvatures

        if "invalid_value" in kwargs:
            self.__invalid_value = kwargs['invalid_value']
            kwargs.pop('invalid_value')

        if 'path' in kwargs:
            kwargs.pop('path')

        for key in kwargs:
            key1 = '_' + self.__class__.__name__ + '__' + key
            if key1 in self.__dict__:
                self.__setattr__(key1, kwargs[key])
            else:

                self.__setattr__(key, kwargs[key])


    def getValues(self):
        """
        :return: min and max curvature
        """
        # TODO: add condition for normalization (if normalized, the values returned should be normalized)
        return np.vstack((self.k1, self.k2))

    def setPointCurvature(self, idx, values):
        """
        Sets a curvature object to specific points

        :param idx: a list or array of indices (can be only one) for which the saliency values refer
        :param values: the curvature objects to assign

        :type idx: list, np.ndarray, int
        :type values: k1 and k2

        """
        self.__principal_curvatures[idx, :] = values

    def getPointCurvature(self, idx):
        """
        Retrieve the curvature object of a specific point

        :param idx: the point index

        :return: principal curvature values (k1, k2)

        :rtype: float

        """
        return self.__principal_curvatures[idx, :]

    def normalize_values(self, bool):
        """
        A flag whether to normalize values of principal curvature (k1, k2) to [0,1] with 2 being the invalid value

        :param bool: boolean flag for normalization of the values

        :type bool: bool

        """
        self.__normalize = bool

    @property
    def k1(self):
        """
        Maximal principal curvature value
        """
        k1 = np.zeros((1, 1))
        if self.Points:
            k1 = self.__principal_curvatures[:, 0]

        if self.Raster:
            k1 = self.__principal_curvatures[:, :, 0]

        # if flag for normalized value is "True", normalize between -1 and 1 with the invalid value set to 1.5

        if self.__normalize:
            # set values larger or smaller than 3sigma the average to invalid value
            k_tmp = k1.copy()
            k_tmp[np.where(k1 < k1.mean() - k1.std() * 3)] = self.__invalid_value
            k_tmp[np.where(k1 > k1.mean() + k1.std() * 3)] = self.__invalid_value

            # normalize to -1 and 1
            # find min and max without considering the invalid value
            min = k_tmp[np.where(k_tmp != self.__invalid_value)].min()
            max = k_tmp[np.where(k_tmp != self.__invalid_value)].max()

            s = 2 / (max - min)
            k1_normed = s * k1 - (1 + min * s)
            k1_normed[np.where(k_tmp == self.__invalid_value)] = 1.5
            k1 = k1_normed

        return k1

    @property
    def k2(self):
        """
        Minimal principal curvature value
        """
        k2 = np.zeros((1, 1))
        if self.Points:
            k2 = self.__principal_curvatures[:, 1]

        if self.Raster:
            k2 = self.__principal_curvatures[:, :, 1]

        # if flag for normalized value is "True", normalize between 0 and 1 with the invalid value set to 1.5
        if self.__normalize:
            # set values larger or smaller than 3sigma the average to invalid value
            k_tmp = k2.copy()
            k_tmp[np.where(k2 < k2.mean() - k2.std() * 3)] = self.__invalid_value
            k_tmp[np.where(k2 > k2.mean() + k2.std() * 3)] = self.__invalid_value

            # normalize to -1 and 1
            # find min and max without considering the invalid value
            min = k_tmp[np.where(k_tmp != self.__invalid_value)].min()
            max = k_tmp[np.where(k_tmp != self.__invalid_value)].max()

            s = 2 / (max - min)
            k2_normed = s * k2 - (1 + min * s)
            k2_normed[np.where(k_tmp == self.__invalid_value)] = 1.5
            k2 = k2_normed

        return k2

    @property
    def mean_curvature(self):
        '''
        Mean curvature values
        '''
        return (self.k1 + self.k2) / 2

    @property
    def gaussian_curvature(self):
        r'''
        Gaussian curvature

        Computed by

        .. math::

           \kappa_{Gaussian}=\lambda_{\max} \cdot \lambda_{\min}

        '''
        return self.k1 * self.k2

    @property
    def curvadness(self):
        r'''
        Curvadness property defined by

        .. math::

            \kappa_{curvadness}=\sqrt{\frac{\lambda_\max ^2 + \lambda_\min ^2}{2}}

        '''
        return np.sqrt((self.k1 ** 2 + self.k2 ** 2) / 2)

    @property
    def Shape_Index(self):
        '''
        '''
        shapeI = np.zeros(self.k1.shape)
        difZero = np.where(np.abs(self.k1 - self.k2) >= 1e-6)[0]  # doesn't equal zero by threshold

        shapeI[difZero] = (1.0 / np.pi) * np.arctan2((self.k2 + self.k1)[difZero], (self.k2 - self.k1)[difZero])
        return shapeI

    def similarity_curvature(self):
        '''
        calculates similarity curvature (E,H)

        :param k1,k2: principal curvatures (k1>k2)

        :return similarCurv: values of similarity curvature (E,H)
        :return rgb: RGB color for every point

        '''
        #         if 'points' in kwargs and ('k1' not in kwargs and 'k2' not in kwargs):
        #             points = kwargs['points']
        #             curv = np.asarray( map( functools.partial( CurvatureFactory.Curvature_FundamentalForm, points = pointSet, rad = coeff, tree = tree ), pp ) )

        k3 = np.min((np.abs(self.k1), np.abs(self.k2)), 0) / np.max((np.abs(self.k1), np.abs(self.k2)), 0)
        similarCurv = np.zeros((k3.shape[0], 2))
        rgb = np.zeros((k3.shape[0], 3), dtype=np.float32)

        sign_k1 = np.sign(self.k1)
        sign_k2 = np.sign(self.k2)
        signK = sign_k1 + sign_k2

        # (+,0)
        positive = np.where(signK == 2)
        similarCurv[positive[0], 0] = k3[positive]
        rgb[positive[0], 0] = k3[positive]
        # (-,0)
        negative = np.where(signK == -2)
        similarCurv[negative[0], 0] = -k3[negative]
        rgb[negative[0], 1] = k3[negative]

        dif = (np.where(signK == 0))[0]
        valueK = np.abs(self.k1[dif]) >= np.abs(self.k2[dif])
        k2_k1 = np.where(valueK == 1)
        k1_k2 = np.where(valueK == 0)
        # (0,+)
        similarCurv[dif[k2_k1[0]], 1] = (k3[dif[k2_k1[0]]].T)[0]
        rgb[dif[k2_k1[0]], 0:2] = np.hstack((k3[dif[k2_k1[0]]], k3[dif[k2_k1[0]]]))
        # (0,-)
        similarCurv[dif[k1_k2[0]], 1] = -(k3[dif[k1_k2[0]]].T)[0]
        rgb[dif[k1_k2[0]], 2] = (k3[dif[k1_k2[0]]].T)[0]

        return rgb, similarCurv
