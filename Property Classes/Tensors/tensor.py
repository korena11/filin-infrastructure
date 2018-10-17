'''
infragit
reuma\Reuma
10, Jul, 2018

.. note::
     Base on Zachi's implementation

A tensor is constructed to a set of points, either around a specific point or around the centeroid (center of gravity)  
'''

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')
import numpy as np
import numpy.linalg as la
from BaseProperty import BaseProperty
from PointSet import PointSet
from functools import partial
from warnings import warn


class Tensor(BaseProperty):
    """
    Class for representing a segment as a Tensor

    .. warning::

       THIS PROPERTY IS NOT FINISHED
    """
    __refPoint = None
    __covMat = None
    __type = None
    __eigenvalues = None
    __eigenvectors = None
    __lambda3 = None  # originally the smallest eigenvalue
    __plateAxis = None
    __stickAxis = None

    def __init__(self, points, covariance_matrix, ref_point, **kwargs):
        """

        :param points: the points incorporated within the construction of the tensor
        :param covariance_matrix: the covariance matrix of the pointset, as computed about the reference point.
        :param ref_point: index of the reference point according to which the covariance matrix was computed. If (-1)
           the ref_point is the centroid of the pointset

        :type points: PointSet
        :type covariance_matrix: np.array
        :type ref_point: int

        """
        super(Tensor, self).__init__(points)
        self.setValues(covariance_matrix, ref_point)
        self.__tensorAnalysis()

    # --------------------- PROPERTIES -------------------------

    @property
    def reference_point(self):
        """
        The point according to which the tensor was computed
        
        """
        return self.__refPoint
        
    @property
    def covariance_matrix(self):
        """
        The covariance matrix of the pointset, as computed about the reference point.

        """
        return self.__covMat

    @property
    def eigenvalues(self):
        """
        Eigenvalues of the covariance matrix

        """
        return self.__eigenvalues


    @property
    def eigenvectors(self):
        """
        Eigenvectors of the covariance matrix

        """
        return self.__eigenvectors

    @property
    def minimal_eigenvalue(self):
        """
        The original minimal eigenvalue.

        """
        return self.__lambda3

    @property
    def tensorType(self):
        """
        The type of the tensor: 'ball' 'stick' or 'plate'

        :rtype: str
        """
        return self.__type

    @property
    def stick_axis(self):
        """
        If the covariance relates to a stick, its normal is the stick_axis
        """
        return self.__stickAxis
    
    def setValues(self, *args, **kwargs):
        """
        Sets values in the tensor object

        :param covariance_matrix:
        :param refPoint: the point about which the tensor is computed


        **Usage**

        .. code-block:: python

          setValues(covariance_matrix, cog)
        """
        self.__covMat = args[0]
        self.__refPoint = args[1]

        try:
            self.__eigenvalues, self.__eigenvectors = la.eigh(self.covariance_matrix)

        except TypeError:
            print(type(self.covariance_matrix[0, 0]))
            print(self.Points.Size[0])
            print(self.covariance_matrix)

        # eigVals /= sum(eigVals)                # Normalizing the eigenvalues
        self.__eigenvalues[abs(self.eigenvalues) <= 1e-8] = 0

        # Computing the plate parameters defined by the tensor
        self.__plateAxis = self.eigenvectors[:, 0]

        # Computing the stick axis defined by the tensor
        self.__stickAxis = self.eigenvectors[:, 2]

    def __tensorAnalysis(self):

        eigVals = self.eigenvalues
        eigVals[0] = eigVals[0] if eigVals[0] > 1e-8 else 1e-8
        # self.__texture = {
        #     'linearity': (eigVals[2] - eigVals[1]) / eigVals[2],
        #     'planarity': (eigVals[1] - eigVals[0]) / eigVals[2],
        #     'planarity2': (eigVals[1] - eigVals[0]) / eigVals[1],
        #     'sphericity': eigVals[0] / eigVals[2],
        #     'omnivariance': (eigVals[0] * eigVals[1] * eigVals[2]) ** (1.0 / 3.0),
        #     'eigenEntropy': -dot(eigVals, log(eigVals)),
        #     'anisotropy': (eigVals[2] - eigVals[0]) / eigVals[2],
        #     'surfaceVariation': eigVals[0],
        #     'verticality': 1 - abs(dot([0, 0, 1], self.__plateAxis))
        # }

        if (eigVals[2] - eigVals[1]) / eigVals[2] > 0.999:
            # The first eigenvalue is much greater the the second one, the tensor corresponds to a stick form
            self.__type = 'stick'

        elif (eigVals[1] - eigVals[0]) / eigVals[1] > 0.9:
            # The second eigenvalue is much greater the the third one, the tensor corresponds to a plate form
            self.__type = 'plate'

        else:
            # the tensor corresponds to a ball form
            self.__type = 'ball'

    def stickRadius(self):
        """
        If the tensor is of stick type -- this computes its radius

        :return: the stick radius

        :rtype: float

        """
        if self.__type == 'stick':
            deltas = self.Points.ToNumpy() - self.reference_point

            # Compute the stick radius of the tensor
            norms = np.array(list(map(la.norm, deltas)))
            distances = norms * np.sin(
                np.array(list(map(np.arccos, map(partial(np.dot, b=self.stick_axis), deltas) / norms))))

            return np.mean(distances)

        else:
            warn(TypeError, 'This tensor is not a stick')
            return 0

    def sphereRadius(self):
        """
        If the tensor is of sphere type -- this compute its radius

        :return: the sphere length, width and height

        :rtype: dict

        """
        if self.__type == 'ball':
            deltas = self.Points.ToNumpy() - self.reference_point
            # Computing the sphere parameters defined by the tensor
            sphereRadius = np.mean(map(la.norm, deltas))

            rotMat = np.array([self.eigenvectors[:, 1].reshape((-1,)),
                               self.eigenvectors[:, 2].reshape((-1,)),
                               self.eigenvectors[:, 0].reshape((-1,))])

            temp = np.array(map(rotMat.dot, deltas)).T
            length, width, height = temp.max(1) - temp.min(1)
            return {'length': length,
                    'width': width,
                    'height': height,
                    'sphereRadius': sphereRadius}
        else:
            warn(TypeError, 'This tensor is not a ball')
            return 0

    def distanceFromPoint(self, point, tensorType='all', sign=False):
        """
        Computing the distance of a given point from the surface defined by the tensor object

        :param point: A 3D point (ndarray, 1x3)
        :param tensorType: The type of tensor to use for the computation ('stick', 'plate', 'ball' or 'all')

        :type point: np.ndarray
        :type tensorType: str

        :return: The distance of the point from the object

        :rtype: float

        """
        from numpy import arccos, dot, sin
        from numpy.linalg import norm

        if tensorType == 'stick':
            distFromCog = norm(point - self.reference_point, axis=0)
            stickRadius = self.stickRadius()
            offAxisAngle = arccos(dot(point - self.reference_point, self.stick_axis.reshape((-1, 1))) / distFromCog)

            return distFromCog * sin(offAxisAngle) - stickRadius

        elif tensorType == 'plate':
            dist = dot(point - self.reference_point, self.__plateAxis.reshape((-1, 1)))

            return dist if sign else abs(dist)

        elif tensorType == 'ball':
            sphereRadius = self.sphereRadius()['sphereRadius']

            return norm(point - self.reference_point, axis=0) - sphereRadius

        elif tensorType == 'all':
            return min([self.distanceFromPoint(point, 'stick'),
                        self.distanceFromPoint(point, 'plate'),
                        self.distanceFromPoint(point, 'ball')])
        else:
            return np.nan

    def logNormDistanceFromPoint(self, point, tensorType='all'):
        """

        .. warning::

            Not working, a function is missing

        Computing the lognormal value for a distance of a given point from the tensor

        :param point:  A 3D point (ndarray, 1x3)
        :param tensorType: The type of tensor to use for the computation ('stick', 'plate', 'ball' or 'all')

        :type point: np.ndarray
        :type tensorType: str

        :return:
        """
        dist = self.distanceFromPoint(point, tensorType)
        return self.__lognormalDistribution.cdf(dist)[0]

    def VisualizeTensor(self, color=(255, 0, 0)):
        from PointSet import PointSet
        from Visualization import Visualization
        from NormalsProperty import NormalsProperty

        pntSet = self.Points
        fig = Visualization.RenderPointSet(pntSet, 'color', pointSize=3, color=color)

        cogPntSet = PointSet(self.reference_point.reshape((1, -1)))
        normalProperty1 = NormalsProperty(cogPntSet, self.tensorStickAxis.reshape((1, -1)) * 10)
        normalProperty2 = NormalsProperty(cogPntSet, self.tensorPlateAxis.reshape((1, -1)) * 10)
        normalProperty3 = NormalsProperty(cogPntSet, self.eigenvectors[:, 1].reshape((1, -1)) * 10)

        Visualization.RenderPointSet(normalProperty1, 'color', color=(0, 0, 255), pointSize=5, _figure=fig)
        Visualization.RenderPointSet(normalProperty2, 'color', color=(0, 0, 255), pointSize=5, _figure=fig)
        Visualization.RenderPointSet(normalProperty3, 'color', color=(0, 0, 255), pointSize=5, _figure=fig)
