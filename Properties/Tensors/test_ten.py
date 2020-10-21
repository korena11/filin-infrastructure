from unittest import TestCase


class TestTensor(TestCase):
    @staticmethod
    def __CreateRandomPointsForPlate(numPnts=100):
        """
        Creating a series of random points for creating random plates
        :param numPnts: Total number of points to generate

        :return: 2D random coordinates
        """
        from numpy.random import random
        x = random((numPnts,))
        y = random((numPnts,))
        return x, y

    @staticmethod
    def __CheckPlate(tensor, normalVector, eps=1e-8, **kwargs):
        """
        Check if the given tensor is a plate with the same normal vector as the given one

        :param tensor: A TensorSegment object to be tested
        :param normalVector: An ndarray representing the normal vector to compare with
        :param eps: The maximum size of the dot product allowed between the vectors
        :return: True if the tensor is a plae with the same normal vector, otherwise false
        """
        tensorParams = tensor.tensorParameters
        if tensor.tensorType != 'plate':
            return False
        elif abs(abs(dot(normalVector, tensorParams[1])) - 1.0) > eps:
            return False
        elif 'x' in kwargs:
            if abs(kwargs['x'] - tensorParams[0][0]) > eps:
                return False
        elif 'y' in kwargs:
            if abs(kwargs['y'] - tensorParams[0][1]) > eps:
                return False
        elif 'z' in kwargs:
            if abs(kwargs['z'] - tensorParams[0][2]) > eps:
                return False
        return True

    def test_reference_point(self):
        self.fail()

    def test_covariance_matrix(self):
        self.fail()

    def test_eigenvalues(self):
        self.fail()

    def test_eigenvectors(self):
        self.fail()

    def test_stick_axis(self):
        self.fail()

    def test_plate_axis(self):
        self.fail()

    def test_points_number(self):
        self.fail()

    def test_setValues(self):
        self.fail()

    def test_distanceFromPoint(self):
        self.fail()

    def test_VisualizeTensor(self):
        self.fail()
