import numpy as np
from scipy.ndimage import gaussian_filter

import RotationUtils
from BaseProperty import BaseProperty
from PointNeighborhood import PointNeighborhood


class NeighborsProperty(BaseProperty):
    """
    Property that holds the neighbors of each point
    """

    def __init__(self, points, *args):
        """

        :param points:

        :type points: PointSet, PointSetOpen3D
        :param args:


        .. warning::
            Now works only for Linux (or with open3D for windows)
        """
        super(NeighborsProperty, self).__init__(points)
        self.__pointsNeighborsArray = np.empty(shape=(self.Size,), dtype=PointNeighborhood)

        # --------- To make the object iterable ---------
        self.current = 0

    # ---------- Definitions to make iterable -----------
    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        try:
            return self.getNeighbors(self.current - 1)
        except IndexError:
            self.current = 0
            raise StopIteration

    # --------end definitions for iterable object-----------
    def GetAllPointsNeighbors(self):
        """
        All neighbors of all computed points
        :return:
        """
        return self.__pointsNeighborsArray

    def getNeighbors(self, idx):
        """
        Retrieve a neighborhood of point(s) with idx index

        :param idx: the index of the point

        :type idx: int, tuple, list

        :return: the tensor of point idx

        :rtype: PointNeighborhood
        """
        neighbors = self.__pointsNeighborsArray[idx]

        if neighbors is None:
            return neighbors

        elif np.all(neighbors.neighborhoodIndices == idx):
            neighbors = None

        return neighbors

    def setNeighbor(self, idx, point_neighbor):
        """
        Set a PointNeighborhood into the property according to the point index

        :param idx: the index (or indices) of the point(s) to set
        :param point_neighbor: the PointeNeighborhood object to set

        :type idx: int
        :type point_neighbor: PointNeighborhood

        """
        self.__pointsNeighborsArray[idx] = point_neighbor

    def RotatePointNeighborhood(self, pointIndex, smoothen=False, useOriginal=False):
        """

        :param pointIndex:
        :param smoothen:
        :param useOriginal:

        :type pointset_open3d: PointSetOpen3D

        :return:
        """
        from PointSetOpen3D import PointSetOpen3D

        if isinstance(self.Points, PointSetOpen3D):
            pointset_open3d = self.Points
        else:
            pointset_open3d = PointSetOpen3D(self.Points)

        pointCoordinates = pointset_open3d.GetPoint(pointIndex)
        pointNeighborhoodPointIdx = self.getNeighbors(pointIndex).GetNeighborhoodIndices()

        if not useOriginal:
            pointNeighborhoodPoints = np.asarray(pointset_open3d.points)[pointNeighborhoodPointIdx]
        else:
            pointNeighborhoodPoints = np.asarray(pointset_open3d.originalPointsOpen3D.points)[pointNeighborhoodPointIdx]

        pointNeighborhoodDiff = pointNeighborhoodPoints - pointCoordinates

        pointNormal = pointset_open3d.normals[pointIndex]
        zAxis = np.array([0., 0., 1.])
        rotationMatrix = RotationUtils.Rotation_2Vectors(pointNormal, zAxis)

        pointNeighborhoodDiff = (np.dot(rotationMatrix, pointNeighborhoodDiff.T)).T
        if smoothen:
            pointNeighborhoodDiff[:, 2] = gaussian_filter(pointNeighborhoodDiff[:, 2], 5)

        self.getNeighbors(pointIndex).SetLocalRotatedNeighbors(pointNeighborhoodDiff)
