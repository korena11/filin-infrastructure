# General Imports
import numpy as np

import GeneralUtils
from PointSet import PointSet
# Infrastructure Imports
from PointSetOpen3D import PointSetOpen3D


class Curve(object):
    def __init__(self, curve_id, points):
        # General Parameters
        self.curve_id = curve_id
        self.points = self.InitializePoints(points)

        self.cdf = None
        self.length = 0
        self.start_end_straight_distance = 0
        self.is_closed = False

        # Fernet-Serret framework
        self.T = np.zeros(len(self.points), dtype=np.ndarray)
        self.N = np.zeros(len(self.points), dtype=np.ndarray)
        self.B = np.zeros(len(self.points), dtype=np.ndarray)

        self.curvature = np.zeros(len(self.points), dtype=np.ndarray)

        # EigenValues, EigenVectors and Points Curvatures
        self.eigen_values = None
        self.eigen_vectors = None
        self.eigen_values_inverse = None
        self.eigen_vectors_inverse = None

        # Preparation Functions
        self.CalculateCDF()
        self.CalculateStartEndStraightDistance()
        # self.CalculatePCA()

    def InitializePoints(self, points):
        '''
        Builds the curve points from numpy array, PointSet or PointSetOpen3D.rst.

        :param points: Curve Points (ordered)
        :return: Curve points as numpy array
        '''
        if isinstance(points, np.ndarray):
            return points
        elif isinstance(points, PointSet) or isinstance(points, PointSetOpen3D):
            return points.ToNumpy()
        else:
            raise TypeError("Curve points can only be given as np.ndArray/PointSet/PointSetOpen3D.rst.")

    def CalculateCDF(self):
        '''
        Calculates the curve's CDF
        :return: None
        '''
        dist = np.linalg.norm(self.points[0:-1, :] - self.points[1:, :], axis=1)
        self.cdf = np.hstack((0, np.cumsum(dist)))  # First value = 0, Last value = curve's length
        self.length = self.cdf[-1]

    def CalculateStartEndStraightDistance(self):
        """


        """
        self.start_end_straight_distance = np.linalg.norm(self.points[0] - self.points[-1])

    def CalculatePCA(self):
        self.eigen_values, self.eigen_vectors = GeneralUtils.PCA(self.points - self.points[0, :])

    def CalculateTangentAndNormalVectors(self, minEuclideanDistance=-1, verbose=False):
        '''
        :param minEuclideanDistance: Minimum distance constraint between selected points for derivative calculations.
        :return:
        '''

        if minEuclideanDistance < 0:
            print("Given Euclidean distance is negative, direct neighbors will be used for derivatives calculations.")

        for currentPointIndex in range(len(self.points)):

            currentCDF = self.cdf[currentPointIndex]
            nextPointIndex = self.__GetNextPointIndex(currentPointIndex, minEuclideanDistance)
            previousPointIndex = self.__GetPreviousPointIndex(currentPointIndex, minEuclideanDistance)

            # Central Derivatives Calculations
            if nextPointIndex and previousPointIndex:
                self.T[currentPointIndex] = self.__CalculateCentralFirstDerivative(nextPointIndex, previousPointIndex)
                self.N[currentPointIndex] = self.__CalculateCentralSecondDerivative(currentPointIndex, nextPointIndex,
                                                                                    previousPointIndex)

            # Forward Derivatives Calculations
            elif nextPointIndex:
                self.T[currentPointIndex] = self.__CalculateForwardFirstDerivative(currentPointIndex, nextPointIndex)
                nextNextPointIndex = self.__GetNextPointIndex(nextPointIndex, minEuclideanDistance)
                if nextNextPointIndex:
                    self.N[currentPointIndex] = self.__CalculateForwardSecondDerivative(currentPointIndex,
                                                                                        nextPointIndex,
                                                                                        nextNextPointIndex)

            # Backward Derivatives Calculations
            elif previousPointIndex:
                self.T[currentPointIndex] = self.__CalculatebackwardFirstDerivative(currentPointIndex,
                                                                                    previousPointIndex)
                previousPreviousPointIndex = self.__GetPreviousPointIndex(previousPointIndex, minEuclideanDistance)
                if previousPreviousPointIndex:
                    self.N[currentPointIndex] = self.__CalculateBackwardSecondDerivative(currentPointIndex,
                                                                                         previousPointIndex,
                                                                                         previousPreviousPointIndex)

            # No Valid Derivatives Calculations
            else:
                # Frenet not found! "Knowledge is power." by: France is Bacon.
                self.T[currentPointIndex] = None
                self.N[currentPointIndex] = None
                self.B[currentPointIndex] = None
                self.curvature[currentPointIndex] = None

        self.CalculateCurvatureAndUnitBinormalVector()

    def __GetNextPointIndex(self, currentPointIndex, minEuclideanDistance):
        '''
        Given a current point index and a minimum euclidean distance, find the next point that satisfies the minimum distance condition.
        If minEuclideanDistance<0 then the next point will be returned no matter it's distance.

        :param currentPointIndex: Index of the current point
        :param minEuclideanDistance: Min distance condition for next point

        :type currentPointIndex: Int
        :type minEuclideanDistance: Float

        :return: Index of the next point if exists and valid. Otherwise, None.
        '''
        if minEuclideanDistance < 0:
            nextPointIndex = currentPointIndex + 1
            if currentPointIndex < nextPointIndex < len(self.points):  # Validate Next Point Index
                return nextPointIndex
            return None

        currentCDF = self.cdf[currentPointIndex]
        nextPointIndex = np.searchsorted(self.cdf, currentCDF + minEuclideanDistance, side='right')

        while True:
            if currentPointIndex < nextPointIndex < len(self.points):  # Validate Next Point Index
                return nextPointIndex
            else:  # Invalid Next Point Index
                nextPointIndex -= 1
                if nextPointIndex == currentPointIndex:
                    return None

    def __GetPreviousPointIndex(self, currentPointIndex, minEuclideanDistance):
        '''
        Given a current point index and a minimum euclidean distance, find the previous point that satisfies the minimum
        distance condition.
        If minEuclideanDistance<0 then the next point will be returned no matter it's distance.

        :param currentPointIndex: Index of the current point
        :param minEuclideanDistance: Min distance condition for next point

        :type currentPointIndex: Int
        :type minEuclideanDistance: Float

        :return: Index of the previous point if exists and valid. Otherwise, None.
        '''
        if minEuclideanDistance < 0:
            previousPointIndex = currentPointIndex - 1
            if currentPointIndex > previousPointIndex > 0:  # Validate Next Point Index
                return previousPointIndex
            return None

        currentCDF = self.cdf[currentPointIndex]
        previousPointIndex = np.searchsorted(self.cdf, currentCDF - minEuclideanDistance, side='left')

        while True:
            if currentPointIndex > previousPointIndex > 0:  # Validate Next Point Index
                return previousPointIndex
            else:  # Invalid Next Point Index
                previousPointIndex += 1
                if previousPointIndex == currentPointIndex:
                    return None

    # region First Derivatives

    def __CalculateForwardFirstDerivative(self, currentPointIndex, nextPointIndex):
        r'''
        Calculate the First Order Forward Derivative.
        Finite Differences Wiki - https://en.wikipedia.org/wiki/Finite_difference

        .. math::

            \bf{f'(x)} = \frac{\bf{f(x + h)} - \bf{f(x)}}{h}

        :param currentPointIndex: Index of the point to calculate the derivative at.
        :param nextPointIndex: Index of the point to calculate the forward derivative towards.

        :type currentPointIndex: Int
        :type nextPointIndex: Int

        :return: Normalized Tangent Vector
        :rtype: NumPy ndArray
        '''

        currentPoint = self.points[currentPointIndex]
        nextPoint = self.points[nextPointIndex]
        cdfCurrentNext = self.cdf[nextPointIndex] - self.cdf[currentPointIndex]

        tangentVector = (nextPoint - currentPoint)
        tangentVector = tangentVector / cdfCurrentNext

        return tangentVector

    def __CalculatebackwardFirstDerivative(self, currentPointIndex, previousPointIndex):
        r'''
        Calculate the First Order Backward Derivative.
        Finite Differences Wiki - https://en.wikipedia.org/wiki/Finite_difference

        .. math::

            \bf{f'(x)} = \frac{\bf{f(x)} - \bf{f(x - h)}}{h}

        :param currentPointIndex: Index of the point to calculate the derivative at.
        :param previousPointIndex: Index of the point to calculate the backward derivative towards.

        :type currentPointIndex: Int
        :type previousPointIndex: Int

        :return: Normalized Tangent Vector
        :rtype: NumPy ndArray
        '''
        currentPoint = self.points[currentPointIndex]
        preivousPoint = self.points[previousPointIndex]
        cdfPreviousCurrent = self.cdf[currentPointIndex] - self.cdf[previousPointIndex]

        tangentVector = (currentPoint - preivousPoint)
        tangentVector = tangentVector / cdfPreviousCurrent

        return tangentVector

    def __CalculateCentralFirstDerivative(self, nextPointIndex, previousPointIndex):
        r'''
        Calculate the First Order Central Derivative.
        Finite Differences Wiki - https://en.wikipedia.org/wiki/Finite_difference

        .. math::

            \bf{f'(x)} = \frac{\bf{f(x + h)} - \bf{f(x - h)}}{2\cdot h}

        :param nextPointIndex: Index of the next point.
        :param previousPointIndex: Index of the previous point.

        :type nextPointIndex: Int
        :type previousPointIndex: Int

        :return: Normalized Tangent Vector
        :rtype: NumPy ndArray
        '''
        nextPoint = self.points[nextPointIndex]
        previousPoint = self.points[previousPointIndex]
        cdfPreviousNext = self.cdf[nextPointIndex] - self.cdf[previousPointIndex]

        tangentVector = (nextPoint - previousPoint)
        tangentVector = tangentVector / cdfPreviousNext

        # tangentVectorNormalized = tangentVector / np.linalg.norm(tangentVector)
        print(np.linalg.norm(tangentVector))

        return tangentVector

    # endregion First Derivatives

    # region Second Derivatives

    def __CalculateCentralSecondDerivative(self, currentPointIndex, nextPointIndex, previousPointIndex):
        r'''
        Calculate the Second Order Central Derivative.
        Finite Differences Wiki - https://en.wikipedia.org/wiki/Finite_difference

        .. math::

            \mathit{f''(x)} = \frac{\mathit{f(x + h)} - \bf{2\mathit{f(x)}} + \mathit{f(x - h))}}{\mathit{h^{2}}}

        :param currentPointIndex: Index of the point to calculate the second derivative at.
        :param nextPointIndex: Index of the next point
        :param previousPointIndex: Index of the previous point.

        :param currentPointIndex: Int
        :type nextPointIndex: Int
        :type previousPointIndex: Int

        :return: Normal Vector
        :rtype: NumPy ndArray
        '''
        currentPoint = self.points[currentPointIndex]
        nextPoint = self.points[nextPointIndex]
        previousPoint = self.points[previousPointIndex]

        cdfCentralAverage = (self.cdf[nextPointIndex] - self.cdf[previousPointIndex]) / 2.

        normalVector = nextPoint - 2. * currentPoint + previousPoint
        normalVector = normalVector / (cdfCentralAverage ** 2)

        return normalVector

    def __CalculateForwardSecondDerivative(self, currentPointIndex, nextPointIndex, nextNextPointIndex):
        r'''
        Calculate the Second Order Forward Derivative.
        Finite Differences Wiki - https://en.wikipedia.org/wiki/Finite_difference

        .. math::

            \mathit{f''(x)} = \frac{\mathit{f(x + \bf{2}\mathit{h})} - \bf{2\mathit{f(x + h)}} + \mathit{f(x))}}{\mathit{h^{2}}}

        :param currentPointIndex: Index of the point to calculate the second derivative at.
        :param nextPointIndex: Index of the next point (at x+h)
        :param nextNextPointIndex: Index of the second next point (at x+2h)

        :param currentPointIndex: Int
        :type nextPointIndex: Int
        :type nextNextPointIndex: Int

        :return: Normal Vector
        :rtype: NumPy ndArray
        '''
        currentPoint = self.points[currentPointIndex]
        nextPoint = self.points[nextPointIndex]
        nextNextPoint = self.points[nextNextPointIndex]

        cdfNextNext = self.cdf[nextNextPointIndex]
        cdfCurrent = self.cdf[currentPointIndex]
        cdfForwardAverage = (cdfNextNext - cdfCurrent) / 2.

        normalVector = nextNextPoint - 2. * nextPoint + currentPoint
        normalVector = normalVector / (cdfForwardAverage ** 2)

        return normalVector

    def __CalculateBackwardSecondDerivative(self, currentPointIndex, previousPointIndex, previousPreviousPointIndex):
        r'''
        Calculate the Second Order Forward Derivative.
        Finite Differences Wiki - https://en.wikipedia.org/wiki/Finite_difference

        .. math::

            \mathit{f''(x)} = \frac{\mathit{f(x)} - \bf{2\mathit{f(x - h)}} + \mathit{f(x - \bf{2} \mathit{h})}}{\mathit{h^{2}}}

        :param currentPointIndex: Index of the point to calculate the second derivative at.
        :param previousPointIndex: Index of the previous point (at x-h)
        :param previousPreviousPointIndex: Index of the second previous point (at x-2h)

        :param currentPointIndex: Int
        :type previousPointIndex: Int
        :type previousPreviousPointIndex: Int

        :return: Normal Vector
        :rtype: NumPy ndArray
        '''
        currentPoint = self.points[currentPointIndex]
        previousPoint = self.points[previousPointIndex]
        previousPreviousPoint = self.points[previousPreviousPointIndex]

        cdfBackwardAverage = (self.cdf[currentPointIndex] - self.cdf[previousPreviousPointIndex]) / 2.

        normalVector = currentPoint - 2. * previousPoint + previousPreviousPoint
        normalVector = normalVector / (cdfBackwardAverage ** 2)

        return normalVector

    # endregion Second Derivatives

    def CalculateCurvatureAndUnitBinormalVector(self):
        for currentIndex in range(len(self.T)):
            currentUnitTangent = self.T[currentIndex]
            if self.T[currentIndex] is not None:
                # Calculating Curvature and Normalizing Normal Vector
                self.curvature[currentIndex] = np.linalg.norm(self.N[currentIndex])
                self.N[currentIndex] = self.N[currentIndex] / self.curvature[currentIndex]

                # Calculating Unit Binormal Vector
                currentUnitNormal = self.N[currentIndex]
                self.B[currentIndex] = np.cross(currentUnitTangent, currentUnitNormal)
                assert np.linalg.norm(self.B[currentIndex]) > 0.995

    def Visualize(self, vectors):
        '''
        ## Temporary Implementation, will be moved to Visualization based Open3D.

        Drawing the desired vector. Possible values:
        'T' : Tangents
        'N' : Normals
        'B' : Bitangents

        **Developer Notes**

        To draw all three vectors at once for each point, should use something like this:
        draw_geometries([point_cloud, line_set])

        :param vectors: Either 'T', 'N' or 'B'/

        :return: None
        '''
        V = getattr(self, vectors)

        colors = np.zeros((len(self.points), 3))
        colors[0] = np.array([0., 1., 0.])
        colors[-1] = np.array([1., 0., 0.])

        import open3d
        PC = open3d.PointCloud()
        PC.points = open3d.Vector3dVector(self.points)
        PC.normals = open3d.Vector3dVector(V)
        PC.colors = open3d.Vector3dVector(colors)
        open3d.draw_geometries([PC])


if __name__ == '__main__':
    import math
    from math import pi


    def points_on_circumference(center, r, n=100):
        return [
            (
                center[0] + (math.cos(2 * pi / n * x) * r),  # x
                center[1] + (math.sin(2 * pi / n * x) * r)  # y

            ) for x in np.arange(0, n + 1)]


    radius = 2.
    numberOfSamples = 100

    sampledPoints = np.array(points_on_circumference(center=(0, 0), r=radius, n=numberOfSamples))
    tempZeros = np.zeros((sampledPoints.shape[0], sampledPoints.shape[1] + 1))
    tempZeros[:, :2] = sampledPoints
    sampledPoints = tempZeros

    halfSampled = np.array_split(sampledPoints, 2)[0]
    # print(np.shape(halfSampled))
    curve = Curve(curve_id=1, points=halfSampled)
    curve.CalculateTangentAndNormalVectors()

    pointsNormals = curve.N
    tempNormal = pointsNormals[0]
    # print(np.linalg.norm(tempNormal))

    curve.Visualize(vectors='N')
