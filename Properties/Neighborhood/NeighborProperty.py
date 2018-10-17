import numpy as np
from scipy.ndimage import gaussian_filter

import RotationUtils
from BaseProperty import BaseProperty
from PointNeighborhood import PointNeighborhood
from PointSet import PointSet
from PointSetOpen3D import PointSetOpen3D


class NeighborsProperty(BaseProperty):
    """
    Property that holds the neighbors of each point
    """

    def __init__(self, points, *args):
        """

        :param points:

        :type points: PointSet
        :param args:


        .. warning::
            Now works only for Linux (or with open3D for windows)
        """
        super(NeighborsProperty, self).__init__(points)
        self.pointsNeighborsArray = np.empty(shape=(self.Size,), dtype=PointNeighborhood)

    def GetPointsNeighborsByID(self, pointset_open3d, idx, searchRadius, maxNN, returnValues=True, override=False,
                               useOriginal=False):
        """
        Get the neighbors of a point within a point cloud by its index.

        :param pointset_open3d: the point set on which the neighbors are looked for
        :param idx: point index
        :param searchRadius: the search radius for neighbors
        :param maxNN: maximum number of neighbors
        :param returnValues: default: True
        :param override: default: False
        :param useOriginal: default: False

        :type pointset_open3d: PointSetOpen3D
        :type idx: int
        :type searchRadius: float
        :type maxNN: int
        :type returnValues: bool
        :type override: bool
        :type useOriginal: bool

        :return: the neighborhood property

        :rtype: NeighborsProperty
        """
        if isinstance(idx, int):
            idx = [idx]

        if override:
            self.__PrintOverrideNeighborhoodCalculations(idx[0], searchRadius, maxNN)

        for currentPointIndex in idx:
            if not override:
                if self.pointsNeighborsArray[currentPointIndex]:
                    r = self.pointsNeighborsArray[currentPointIndex].GetRadius()
                    nn = self.pointsNeighborsArray[currentPointIndex].GetMaxNN()
                    if r == searchRadius and nn == maxNN:
                        continue

            currentPoint = pointset_open3d.GetPoint(currentPointIndex)

            # currentPoint = self.pointsOpen3D.points[currentPointIndex]
            pointNeighborhoodObject = self.GetPointNeighborsByCoordinates(point=currentPoint, searchRadius=searchRadius,
                                                                          maxNN=maxNN, useOriginal=useOriginal)
            self.pointsNeighborsArray[currentPointIndex] = pointNeighborhoodObject
            self.__RotatePointNeighborhood(currentPointIndex, smoothen=False, useOriginal=useOriginal)

        if returnValues:
            if len(idx) == 1:
                return self.pointsNeighborsArray[idx][0]
            return self.pointsNeighborsArray[idx]

    def GetPointNeighborsByCoordinates(self, pointset_open3d, point, searchRadius, maxNN, useOriginal=False):
        if maxNN <= 0:
            if not useOriginal:
                num, idx, dist = pointset_open3d.kdTreeOpen3D.search_radius_vector_3d(point, radius=searchRadius)
            else:
                num, idx, dist = pointset_open3d.originalkdTreeOpen3D.search_radius_vector_3d(point,
                                                                                              radius=searchRadius)

        elif searchRadius <= 0:
            if not useOriginal:
                num, idx, dist = pointset_open3d.kdTreeOpen3D.search_knn_vector_3d(point, knn=maxNN)
            else:
                num, idx, dist = pointset_open3d.originalkdTreeOpen3D.search_knn_vector_3d(point, knn=maxNN)

        else:
            if not useOriginal:
                num, idx, dist = pointset_open3d.kdTreeOpen3D.search_hybrid_vector_3d(point, radius=searchRadius,
                                                                                      max_nn=maxNN)
            else:
                num, idx, dist = pointset_open3d.originalkdTreeOpen3D.search_hybrid_vector_3d(point,
                                                                                              radius=searchRadius,
                                                                                              max_nn=maxNN)

        pointNeighborhood = PointNeighborhood(searchRadius, maxNN, num, idx, dist)
        return pointNeighborhood

    def __RotatePointNeighborhood(self, pointset_open3d, pointIndex, smoothen=False, useOriginal=False):
        """

        :param pointset_open3d:
        :param pointIndex:
        :param smoothen:
        :param useOriginal:

        :type pointset_open3d: PointSetOpen3D

        :return:
        """
        pointCoordinates = pointset_open3d.GetPoint(pointIndex)
        pointNeighborhoodPointIdx = self.pointsNeighborsArray[pointIndex].GetNeighborhoodIndices()

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

        self.pointsNeighborsArray[pointIndex].SetLocalRotatedNeighbors(pointNeighborhoodDiff)

    def GetAllPointsNeighbors(self):
        """
        All neighbors of all computed points
        :return:
        """
        return self.pointsNeighborsArray

    def __PrintOverrideNeighborhoodCalculations(self, exampleIndex, newRadius, newMaxNN):
        """

        :param exampleIndex:
        :param newRadius:
        :param newMaxNN:
        :return:
        """
        previousRadius = self.pointsNeighborsArray[exampleIndex].GetRadius()
        previousMaxNN = self.pointsNeighborsArray[exampleIndex].GetMaxNN()

        if previousRadius != newRadius or previousMaxNN != newMaxNN:
            print("Function: PointSetOpen3D.PointSetOpen3D.GetPointsNeighborsByID")
            print("Overriding Previous Calculations")

            print("Previous Radius/maxNN: " + str(previousRadius) + "/" + str(previousMaxNN))
            print("New Radius/maxNN:\t" + str(newRadius) + "/" + str(newMaxNN))
            print()
