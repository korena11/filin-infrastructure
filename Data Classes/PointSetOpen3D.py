# Utils Imports
# General Imports
import numpy as np
# 3rd Party Imports
import open3d as O3D
from scipy.ndimage import gaussian_filter

import RotationUtils
from PointNeighborhood import PointNeighborhood
# Infrastructure Imports
from PointSet import PointSet
from RandomColors import LetThereBeRandomColors

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

class PointSetOpen3D(PointSet):
    def __init__(self, inputPoints):

        self.pointSet = super(PointSetOpen3D, self).__init__(inputPoints)

        self.pointsOpen3D = None
        self.originalPointsOpen3D = None  # Will be useful only if down sampling was performed. This stores the original

        self.disregardingMasksList = []

        self.voxelSize = 0.

        self.InitializeOpen3dObject(inputPoints)

        # TODO: check if needed after Elia
        self.pointsNeighborsArray = np.empty(shape=(self.Size,), dtype=PointNeighborhood)
        self.originalNumberOfPoints = len(self.pointsOpen3D.points)

        self.kdTreeOpen3D = O3D.KDTreeFlann(self.pointsOpen3D)
        self.originalkdTreeOpen3D = O3D.KDTreeFlann(self.originalPointsOpen3D)

    def InitializeOpen3dObject(self, inputPoints):
        """
        Initializes the object according to the type of the input points

        :param inputPoints: the points from which the object should be initialized to

        :type inputPoints: np.ndarray, o3D.PointCloud, PointSet
        :return:
        """
        if isinstance(inputPoints, np.ndarray):
            self.pointsOpen3D = O3D.PointCloud()
            self.pointsOpen3D.points = O3D.Vector3dVector(inputPoints)
        elif isinstance(inputPoints, PointSet):
            self.pointsOpen3D = O3D.PointCloud()
            pts = inputPoints.ToNumpy()[:, :3]
            self.pointsOpen3D.points = O3D.Vector3dVector(pts)

        elif isinstance(inputPoints, O3D.PointCloud):
            self.pointsOpen3D = inputPoints
        else:
            print("Given type: " + str(type(inputPoints)) + " as input. Not sure what to do with that...")
            raise ValueError("Wrong turn.")

        self.originalPointsOpen3D = O3D.PointCloud(self.pointsOpen3D)
        print(self.originalPointsOpen3D)

    def RebuildKDTree(self, verbose=True):
        if verbose:
            print("Rebuilding KDTree...")
        self.kdTreeOpen3D = O3D.KDTreeFlann(self.pointsOpen3D)

    # def GetPoint(self, indx=None):
    #     pointsArray = np.asarray(self.pointsOpen3D.points)
    #     if indx:
    #         pointsArray = pointsArray[indx]
    #
    #     return pointsArray

    def ToNumpy(self):
        pointsArray = np.asarray(self.originalPointsOpen3D.points)
        return pointsArray

    @property
    def Size(self):
        return len(self.pointsOpen3D.points)

    @Size.setter
    def Size(self, new_size):
        self.Size = new_size

    def DownsampleCloud(self, voxelSize, verbose=True):
        if voxelSize > 0.:
            self.pointsOpen3D = O3D.voxel_down_sample(self.pointsOpen3D, voxel_size=voxelSize)

            if verbose:
                print("Downsampling the point cloud with a voxel size of " + str(voxelSize))
                print("Number of points after down sampling: " + str(self.pointsOpen3D))

            self.voxelSize = voxelSize
            self.numberOfPoints = len(self.pointsOpen3D.points)
            self.pointsNeighborsArray = np.empty(shape=(self.numberOfPoints,), dtype=PointNeighborhood)
            self.RebuildKDTree()

    def CalculateNormals(self, searchRadius=0.05, maxNN=20, orientation=(0., 0., 0.), verbose=True):
        if verbose:
            print(">>> Calculating point-cloud normals. Neighborhood Parameters -- r:" + str(
                searchRadius) + "\tnn:" + str(
                maxNN))

        if maxNN <= 0:
            O3D.estimate_normals(self.pointsOpen3D,
                                 search_param=O3D.KDTreeSearchParamRadius(radius=searchRadius))
        elif searchRadius <= 0:
            O3D.estimate_normals(self.pointsOpen3D,
                                 search_param=O3D.KDTreeSearchParamKNN(knn=maxNN))
        else:
            O3D.estimate_normals(self.pointsOpen3D,
                                 search_param=O3D.KDTreeSearchParamHybrid(radius=searchRadius, max_nn=maxNN))

        if isinstance(orientation, tuple):
            if orientation == (0., 0., 0.):
                O3D.orient_normals_towards_camera_location(self.pointsOpen3D)  # Default Camera Location is (0, 0, 0).
            else:
                raise NotImplementedError("Need to modify...")
                O3D.orient_normals_to_align_with_direction(self.pointsOpen3D)  # Default Direction is (0, 0, 1).
        else:
            raise ValueError("Orientation should be a tuple representing a location (X, Y, Z).\n"
                             "Default Location: Camera (0., 0., 0.).")

    def DisregardOtherPoints(self, indx):
        """
        Remove points from the PointSetOpen3D object

        :param indx: indices to remove

        :return:
        """
        if isinstance(indx, int):
            indx = [indx]

        if len(indx) > 0:
            # Update Points
            points = np.asarray(self.pointsOpen3D.points)
            points = points[indx, :]
            self.pointsOpen3D.points = O3D.Vector3dVector(points)
            self.kdTreeOpen3D = O3D.KDTreeFlann(self.pointsOpen3D)

            # Update Points Colors
            pointsColors = np.asarray(self.pointsOpen3D.colors)
            pointsColors = pointsColors[indx, :]
            self.pointsOpen3D.colors = O3D.Vector3dVector(pointsColors)

            # Update Points Normals
            pointsNormals = np.asarray(self.pointsOpen3D.normals)
            if pointsNormals.size:
                pointsNormals = pointsNormals[indx, :]
                self.pointsOpen3D.normals = O3D.Vector3dVector(pointsNormals)

            # Update Neighborhood Array
            self.pointsNeighborsArray = self.pointsNeighborsArray[indx]

            # Update Number of Points
            self.numberOfPoints = len(indx)

            # Rebuild KD Tree
            self.RebuildKDTree()

    def DisregardPoints(self, indx, verbose=True):
        """
        Remove points from the PointSetOpen3D object

        :param indx: indices to remove
        :param verbose: runtime printing. Default: True


        :type indx: list or int
        :type verbose: bool

        """
        if isinstance(indx, int):
            indx = [indx]

        if len(indx) > 0:
            if verbose:
                print("Disregarding " + str(len(indx)) + " points.")

            # Update Points
            points = np.asarray(self.ToNumpy())
            points = np.delete(points, indx, axis=0)
            self.pointsOpen3D.points = O3D.Vector3dVector(points)
            self.kdTreeOpen3D = O3D.KDTreeFlann(self.pointsOpen3D)

            # Update Points Colors
            pointsColors = np.asarray(self.pointsOpen3D.colors)
            pointsColors = np.delete(pointsColors, indx, axis=0)
            self.pointsOpen3D.colors = O3D.Vector3dVector(pointsColors)

            # Update Points Normals
            pointsNormals = np.asarray(self.pointsOpen3D.normals)
            if pointsNormals.size:
                pointsNormals = np.delete(pointsNormals, indx, axis=0)
                self.pointsOpen3D.normals = O3D.Vector3dVector(pointsNormals)

            # Update Neighborhood Array
            self.pointsNeighborsArray = np.delete(self.pointsNeighborsArray, indx, axis=0)

            if verbose:
                print("Number of remaining points: " + str(self.Size))

            # Rebuild KD Tree
            self.RebuildKDTree(verbose=verbose)

    # region Neighborhood Functions

    def GetPointsNeighborsByID(self, idx, searchRadius, maxNN, returnValues=True, override=False,
                               useOriginal=False):
        """

        :param idx: point index
        :param searchRadius: the search radius for neighbors
        :param maxNN: maximum number of neighbors
        :param returnValues: default: True
        :param override: default: False
        :param useOriginal: default: False

        :type idx: int
        :type searchRadius: float
        :type maxNN: int
        :type returnValues: bool
        :type override: bool
        :type useOriginal: bool

        :return:
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
                    if (r == searchRadius and nn == maxNN):
                        continue

            currentPoint = self.pointsOpen3D.points[currentPointIndex]
            pointNeighborhoodObject = self.GetPointNeighborsByCoordinates(point=currentPoint, searchRadius=searchRadius,
                                                                          maxNN=maxNN, useOriginal=useOriginal)
            self.pointsNeighborsArray[currentPointIndex] = pointNeighborhoodObject
            self.__RotatePointNeighborhood(currentPointIndex, smoothen=False, useOriginal=useOriginal)

        if returnValues:
            if len(idx) == 1:
                return self.pointsNeighborsArray[idx][0]
            return self.pointsNeighborsArray[idx]

    def GetPointNeighborsByCoordinates(self, point, searchRadius, maxNN, useOriginal=False):
        if maxNN <= 0:
            if not useOriginal:
                num, idx, dist = self.kdTreeOpen3D.search_radius_vector_3d(point, radius=searchRadius)
            else:
                num, idx, dist = self.originalkdTreeOpen3D.search_radius_vector_3d(point, radius=searchRadius)

        elif searchRadius <= 0:
            if not useOriginal:
                num, idx, dist = self.kdTreeOpen3D.search_knn_vector_3d(point, knn=maxNN)
            else:
                num, idx, dist = self.originalkdTreeOpen3D.search_knn_vector_3d(point, knn=maxNN)

        else:
            if not useOriginal:
                num, idx, dist = self.kdTreeOpen3D.search_hybrid_vector_3d(point, radius=searchRadius, max_nn=maxNN)
            else:
                num, idx, dist = self.originalkdTreeOpen3D.search_hybrid_vector_3d(point, radius=searchRadius,
                                                                                   max_nn=maxNN)

        pointNeighborhood = PointNeighborhood(searchRadius, maxNN, num, idx, dist)
        return pointNeighborhood

    def __RotatePointNeighborhood(self, pointIndex, smoothen=False, useOriginal=False):
        pointCoordinates = self.pointsOpen3D.points[pointIndex]
        pointNeighborhoodPointIdx = self.pointsNeighborsArray[pointIndex].neighborhoodIndices

        if not useOriginal:
            pointNeighborhoodPoints = np.asarray(self.pointsOpen3D.points)[pointNeighborhoodPointIdx]
        else:
            pointNeighborhoodPoints = np.asarray(self.originalPointsOpen3D.points)[pointNeighborhoodPointIdx]

        pointNeighborhoodDiff = pointNeighborhoodPoints - pointCoordinates

        pointNormal = self.pointsOpen3D.normals[pointIndex]
        zAxis = np.array([0., 0., 1.])
        rotationMatrix = RotationUtils.Rotation_2Vectors(pointNormal, zAxis)

        pointNeighborhoodDiff = (np.dot(rotationMatrix, pointNeighborhoodDiff.T)).T
        if smoothen:
            pointNeighborhoodDiff[:, 2] = gaussian_filter(pointNeighborhoodDiff[:, 2], 5)

        self.pointsNeighborsArray[pointIndex].localRotatedNeighbors = pointNeighborhoodDiff

    def GetAllPointsNeighbors(self):
        return self.pointsNeighborsArray

    # endregion

    # region Visualization Functions
    def SetPointsColors(self, colors):
        if isinstance(colors, O3D.Vector3dVector):
            self.pointsOpen3D.colors = colors
        else:
            self.pointsOpen3D.colors = O3D.Vector3dVector(colors)

    def VisualizeClusters(self, pointsLabels, both=False):
        assert self.numberOfPoints == len(pointsLabels), "Dimensions do not match."

        clustersIDs = set(pointsLabels)
        numberOfClusters = len(clustersIDs)

        randomColors = LetThereBeRandomColors().GenerateNewColor(numberOfColorsToGenerate=numberOfClusters)

        for currentInd in range(1, numberOfClusters + 1):
            selectedColor = randomColors[currentInd - 1]
            np.asarray(self.pointsOpen3D.colors)[pointsLabels == currentInd] = selectedColor

        self.Visualize(both=both)

    def Visualize(self, original=False, both=False):
        def toggle_black_white_background(vis):
            opt = vis.get_render_option()
            if np.array_equal(opt.background_color, np.ones(3)):
                opt.background_color = np.zeros(3)
            else:
                opt.background_color = np.ones(3)
            return False

        key_to_callback = {}
        key_to_callback[ord("K")] = toggle_black_white_background

        if both:
            originalColors = np.zeros((self.originalNumberOfPoints, 3), dtype=np.float)
            originalColors[:, 0] = 0.5
            # subsetColors = np.zeros((self.numberOfPoints, 3), dtype=np.float)
            # subsetColors[:, 0] = 1.
            # self.pointsOpen3D.colors = O3D.Vector3dVector(subsetColors)
            self.originalPointsOpen3D.colors = O3D.Vector3dVector(originalColors)
            drawData = [self.pointsOpen3D, self.originalPointsOpen3D]
        elif original:
            drawData = [self.originalPointsOpen3D]
        else:
            drawData = [self.pointsOpen3D]

        O3D.draw_geometries_with_key_callbacks(drawData, key_to_callback)

    # endregion

    def __PrintOverrideNeighborhoodCalculations(self, exampleIndex, newRadius, newMaxNN):
        previousRadius = self.pointsNeighborsArray[exampleIndex].GetRadius()
        previousMaxNN = self.pointsNeighborsArray[exampleIndex].GetMaxNN()

        if previousRadius != newRadius or previousMaxNN != newMaxNN:
            print("Function: PointSetOpen3D.PointSetOpen3D.GetPointsNeighborsByID")
            print("Overriding Previous Calculations")

            print("Previous Radius/maxNN: " + str(previousRadius) + "/" + str(previousMaxNN))
            print("New Radius/maxNN:\t" + str(newRadius) + "/" + str(newMaxNN))
            print()
