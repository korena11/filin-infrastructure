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
from PointSet import PointSet


class Tensor(object):
    """
    Class for representing a segment as a Tensor

    """
    __refPoint = None
    __covMat = None
    __eigenvalues = None
    __eigenvectors = None
    __plateAxis = None
    __stickAxis = None
    __num_points = None

    def __init__(self, covariance_matrix, ref_point, pts_number, **kwargs):
        """

        :param covariance_matrix: the covariance matrix of the pointset, as computed about the reference point.
        :param ref_point: index of the reference point according to which the covariance matrix was computed. If (-1)
           the ref_point is the centroid of the pointset
        :param pts_number: the number of points that were used to compute the tensor

        :type points: PointSet
        :type covariance_matrix: np.array
        :type ref_point: int

        """

        self.setValues(covariance_matrix, ref_point)

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
    def stick_axis(self):
        """
        If the covariance relates to a stick, its normal is the stick_axis
        """
        return self.__stickAxis

    @property
    def plate_axis(self):
        """
        If the covariance relates to a plane (a plate), its normal is the plate_axis

        """
        return self.__plateAxis

    @property
    def points_number(self):
        """
        Number of points used for tensor computation


        """
        return self.__num_points

    # ------------------------- FUNCTIONS -------------------------------

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
            print(self.__num_points)
            print(self.covariance_matrix)

        # eigVals /= sum(eigVals)                # Normalizing the eigenvalues
        self.__eigenvalues[abs(self.eigenvalues) <= 1e-8] = 0

        # Computing the plate parameters defined by the tensor
        self.__plateAxis = self.eigenvectors[:, 0]

        # Computing the stick axis defined by the tensor
        self.__stickAxis = self.eigenvectors[:, 2]


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
            offAxisAngle = arccos(dot(point - self.reference_point, self.stick_axis.reshape((-1, 1))) / distFromCog)

            return distFromCog * sin(offAxisAngle)

        elif tensorType == 'plate':
            dist = dot(point - self.reference_point, self.plate_axis.reshape((-1, 1)))

            return dist if sign else abs(dist)

        elif tensorType == 'ball':
            return norm(point - self.reference_point, axis=0)

        elif tensorType == 'all':
            return min([self.distanceFromPoint(point, 'stick'),
                        self.distanceFromPoint(point, 'plate'),
                        self.distanceFromPoint(point, 'ball')])
        else:
            return np.nan

    def VisualizeTensor(self, pntSet, color=(255, 0, 0)):
        """
        NOT WORKING
        
        :param pntSet:
        :param color:
        :return:
        """
        from PointSet import PointSet
        from VisualizationVTK import VisualizationVTK
        from NormalsProperty import NormalsProperty

        fig = VisualizationVTK.RenderPointSet(pntSet, 'color', pointSize=3, color=color)

        cogPntSet = PointSet(self.reference_point.reshape((1, -1)))
        normalProperty1 = NormalsProperty(cogPntSet, self.stick_axis.reshape((1, -1)) * 10)
        normalProperty2 = NormalsProperty(cogPntSet, self.plate_axis.reshape((1, -1)) * 10)
        normalProperty3 = NormalsProperty(cogPntSet, self.eigenvectors[:, 1].reshape((1, -1)) * 10)

        VisualizationVTK.RenderPointSet(normalProperty1, 'color', color=(0, 0, 255), pointSize=5, _figure=fig)
        VisualizationVTK.RenderPointSet(normalProperty2, 'color', color=(0, 0, 255), pointSize=5, _figure=fig)
        VisualizationVTK.RenderPointSet(normalProperty3, 'color', color=(0, 0, 255), pointSize=5, _figure=fig)
