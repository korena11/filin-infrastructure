import numpy as np
from numpy import array
from sklearn.neighbors import KDTree

from PointSet import PointSet


class KdTreePointSet(PointSet):
    def __init__(self, points, path=None, intensity=None, range_accuracy=0.002, angle_accuracy=0.012,
                 measurement_accuracy=0.002, leaf_size=40, **kwargs):
        r"""
        A kd-tree representation

        :param points: the points to represent as kd-tree
        :param leaf_size: minimal number of points in a leaf, with the maximal :math:`2\cdot` leaf_size

        :type points: PointSet, np.array
        :type leaf_size: int

        """
        super(KdTreePointSet, self).__init__(points, path, intensity, range_accuracy, angle_accuracy,
                                             measurement_accuracy)

        self.__initializeKdTree(points, leaf_size)

    # ---------------- PRIVATES-------------
    def __initializeKdTree(self, inputPoints, leafsize):
        """
        Initialize the object according to the type of the input points

        :param inputPoints: the points from which the object should be initialized to

        :type inputPoints: np.ndarray, o3D.PointCloud, PointSet

        """
        import open3d as O3D

        if isinstance(inputPoints, np.ndarray):
            self.data = KDTree(inputPoints, leafsize)

        elif isinstance(inputPoints, PointSet):
            pts = inputPoints.ToNumpy()[:, :3]
            self.data = KDTree(pts, leafsize)
            self.path = inputPoints.path

        elif isinstance(inputPoints, O3D.PointCloud):
            pts = np.asarray(inputPoints.points)
            self.data = KDTree(pts, leafsize)
        else:
            print("Given type: " + str(type(inputPoints)) + " as input. Not sure what to do with that...")
            raise ValueError("Wrong turn.")

    # ------------- PROPERTIES -----------------
    @property
    def X(self):
        return self.ToNumpy()[:, 0]

    @property
    def Y(self):
        return self.ToNumpy()[:, 1]

    @property
    def Z(self):
        return self.ToNumpy()[:, 2]

    @property
    def Size(self):
        return self.ToNumpy().shape[0]

    # ------------ GENERAL FUNCTIONS------------------
    def query(self, pnts, k):
        """
        Query the kd-tree for the k nearest neighbors of a given set of points

        :param pnts: The query points
        :param k: The number of neighbors to find for the point

        :type pnts: np.array nx3
        :type k: int

        :return: The indexes for the neighbors of the points

        :rtype: list of np.array

        .. note::
            Return the query points themselves as the first index of each list

        """
        distances, indexes = self.data.query(pnts, k=k)
        return indexes

    def queryRadius(self, pnts, radius):
        """
        Query the kd-tree to find the neighbors of a given set of point inside a given radius

        :param pnts: The query points
        :param radius: The query radius

        :type pnts: np.array nx3
        :type radius: float

        :return: The indexes for the neighbors of the points

        :rtype: list of np.array

        .. note::
            Return the query points themselves as the first index of each list

        """
        if isinstance(pnts, list):
            pnts = array(pnts)

        if pnts.ndim == 1:
            indexes = self.data.query_radius(pnts.reshape((1, -1)), radius)

            if indexes.dtype == object:
                indexes = indexes[0]

        else:
            indexes = self.data.query_radius(pnts, radius)

        return indexes

    def ToNumpy(self):
        """
        Points as numpy

        """
        return self.data.get_arrays()[0]

    def GetPoint(self, index):
        return self.ToNumpy()[index, :]

    def ToPolyData(self):
        from VisualizationUtils import MakeVTKPointsMesh
        vtkPolyData = MakeVTKPointsMesh(self.ToNumpy())
        return vtkPolyData

    def save(self, path_or_buf, **kwargs):
        # TODO: IMPLEMENT save to json file
        pass


if __name__ == '__main__':
    from numpy.random import random

    points = random((10000, 3))
    kdTree = KdTreePointSet(points)
    print(kdTree.query([[0.5, 0.5, 0.5], [0.75, 0.75, 0.75]], 7))
    print(kdTree.queryRadius([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], 0.1))
