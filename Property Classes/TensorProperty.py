'''
infragit
reuma\Reuma
10, Jul, 2018 
'''

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

import numpy.linalg as la
from BaseProperty import BaseProperty
from PointSet import PointSet


class TensorProperty(BaseProperty):
    """
    Class for representing a segment as a Tensor

    .. warning::

       THIS PROPERTY IS NOT FINISHED
    """
    __refPoint = None
    __covMat = None
    __stickAxis = None
    __eigenvalues = None
    __eigenvectors = None
    __lambda3 = None  # originally the smallest eigenvalue
    __plateAxis = None

    def __init__(self, points, covariance_matrix, ref_point, **kwargs):
        """

        :param points:
        :param covariance_matrix: the covariance matrix of the pointset, as computed about the reference point.
        :param ref_point: index of the reference point according to which the covariance matrix was computed. If (-1)
           the ref_point is the centroid of the pointset

        :type points: PointSet
        :type covariance_matrix: np.array
        :type ref_point: int

        """
        super(TensorProperty, self).__init__(points)
        self.setValues(covariance_matrix, ref_point)

    @property
    def stick_axis(self):
        """
        If the covariance relates to a stick, its normal is the stick_axis
        """
        return self.__stickAxis

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
        self.__eigenvalues[abs(self.__eigenvalues) <= 1e-8] = 0

        # Computing the plate parameters defined by the tensor
        self.__plateAxis = self.eigenvectors[:, 0]
