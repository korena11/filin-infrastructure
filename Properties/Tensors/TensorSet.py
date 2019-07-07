from numpy import dot, cross, array
from numpy.linalg import norm

from Tensor import Tensor
from TensorFactory import TensorFactory


class TensorSet(object):

    def __init__(self, tensors=None):

        self.__overallTensor = None
        self.__numTensors = 0
        self.__tensors = []

        if tensors is None:
            raise UserWarning('\'tensors\' parameter is None')
        elif isinstance(tensors, Tensor):
            self.__overallTensor = tensors
            self.__numTensors = 1
            self.__tensors.append(tensors)
        else:
            # adding all the tensors to the set and updating the general tensor
            # list(map(self.addTensor, tensors))
            self.__overallTensor = TensorFactory.unifyTensors(tensors)
            self.__numTensors = len(tensors)
            self.__tensors = tensors if isinstance(tensors, list) else tensors.tolist()

            if abs(norm(self.__overallTensor.covariance_matrix)) < 1e-6:
                a = 1

    def addTensor(self, tensor):
        """
        Adding a tensor to the TensorSet
        :param tensor: A tensor object (Tensor)
        :return: None
        """
        if isinstance(tensor, TensorSet):
            # tensor to add is a TensorSet object: adding all the individual tensors
            list(map(self.addTensor, tensor.tensors))

        if self.__overallTensor is None:
            self.__overallTensor = tensor

        else:
            self.__overallTensor = TensorFactory.joinTensors(self.__overallTensor, tensor)

        self.__tensors.append(tensor)
        self.__numTensors += 1

    @property
    def points(self):
        """
        The points that were used to construct the tensor

        :rtype: PointSet, PointSubSet.PointSubSet, BaseData.BaseData

        """
        return self.__overallTensor.points

    @property
    def reference_point(self):
        """
        The point according to which the tensor was computed

        """
        return self.__overallTensor.reference_point

    @property
    def covariance_matrix(self):
        """
        The covariance matrix of the pointset, as computed about the reference point.

        """
        return self.__overallTensor.covariance_matrix

    @property
    def eigenvalues(self):
        """
        Eigenvalues of the covariance matrix

        """
        return self.__overallTensor.eigenvalues

    @property
    def eigenvectors(self):
        """
        Eigenvectors of the covariance matrix

        """
        return self.__overallTensor.eigenvectors

    @property
    def stick_axis(self):
        """
        If the covariance relates to a stick, its normal is the stick_axis
        """
        return self.__overallTensor.stick_axis

    @property
    def plate_axis(self):
        """
        If the covariance relates to a plane (a plate), its normal is the plate_axis
        """
        return self.__overallTensor.plate_axis

    @property
    def points_number(self):
        """
        Number of points used for tensor computation
        """
        return self.__overallTensor.points_number

    @property
    def tensors_number(self):
        """
        Returns the number of tensors used to compute the overall one
        :return: number of tensors (int)
        """
        return self.__numTensors

    @property
    def tensors(self):
        """
        Returns the list of tensors used to computed the overall one
        :return: list of Tensor objects
        """
        return self.__tensors

    def distanceFromPoint(self, pnt, tensorType='plate', sign=False, basedOnOverallTensor=True):
        """
        Computes the distance of a point from the TensorSet. Computation can be done either based on the overall
        tensor or based on the minimum distance from the set of tensors
        :param pnt: point to compute the distance from
        :param tensorType: indicator for choosing which shape to compute the distance by (optional, string)
        :param sign: indicator whether to returned a signed distance or absolute value (optional, bool)
        :param basedOnOverallTensor: indicator whether to compute the distance based on the overall tensor or the
        list of them (optional, bool)
        :return: distance (float)
        """
        if basedOnOverallTensor:
            return self.__overallTensor.distanceFromPoint(pnt, tensorType, sign)
        else:
            refPoints = array(list(map(lambda t: t.referece_point, self.__tensors)))
            deltas = refPoints - pnt.reshape((-1, 3))

            if tensorType == 'stick':
                stickAxes = array(list(map(lambda t: t.stick_axis, self.__tensors)))
                return norm(cross(stickAxes, deltas), axis=1).min()
            elif tensorType == 'plate':
                plateAxes = array(list(map(lambda t: t.plate_axis, self.__tensors)))
                dist = dot(plateAxes, deltas.T).min()
                return abs(dist) if sign else dist
            elif tensorType == 'ball':
                return norm(deltas, axis=1).min()
            else:
                raise ValueError('Unsupported tensor type')
