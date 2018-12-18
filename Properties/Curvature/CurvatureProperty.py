import numpy as np

from BaseProperty import BaseProperty


class CurvatureProperty(BaseProperty):
    '''
    classdocs
    '''

    __curvature = None

    def __init__(self, points, curvature):
        super(CurvatureProperty, self).__init__(points)
        # self._BaseProperty__points = points

        self.setValues(curvature)

    def setValues(self, *args, **kwargs):
        """
        Sets curvature into Curvature Property object

        :param curvature:

        """
        self.__curvature = args[0]

    def getValues(self):
        """
        :return: min and max curvature
        """
        return np.vstack((self.k1, self.k2))

    @property
    def Curvature(self):
        """
        Points' curvature values
        """
        return self.__curvature

    @property
    def k1(self):
        """
        Maximal principal curvature value
        """
        if self.Points:
            return self.__curvature[:, 0]

        if self.Raster:
            return self.__curvature[:, :, 0]

    @property
    def k2(self):
        """
        Minimal principal curvature value
        """
        if self.Points:
            return self.__curvature[:, 1]

        if self.Raster:
            return self.__curvature[:, :, 1]

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
        equalZero = np.where(np.abs(self.k1 - self.k2) <= 1e-6)[0]
        difZero = np.where(self.k1 != self.k2)[0]
        if equalZero.size != 0:
            shapeI[equalZero, :] = 0
        shapeI[difZero, :] = (1.0 / np.pi) * np.arctan2((self.k2 + self.k1)[difZero], (self.k2 - self.k1)[difZero])
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
