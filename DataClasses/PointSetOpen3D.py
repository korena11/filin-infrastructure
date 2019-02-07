# Utils Imports
# General Imports
import numpy as np
# 3rd Party Imports
import open3d as O3D

# from PointNeighborhood import PointNeighborhood
# Infrastructure Imports
from PointSet import PointSet
from RandomColors import LetThereBeRandomColors

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

class PointSetOpen3D(PointSet):

    def __init__(self, points, path=None, intensity=None, range_accuracy=0.002, angle_accuracy=0.012,
                 measurement_accuracy=0.002):

        super(PointSetOpen3D, self).__init__(points, path, intensity, range_accuracy, angle_accuracy,
                                             measurement_accuracy)

        self.InitializeOpen3dObject(points)  # sets the data to be open3d object
        self.voxelSize = 0.
        self.kdTreeOpen3D = O3D.KDTreeFlann(self.data)

    def GetPoint(self, index):
        """
        Retrieve specific point(s) by index

        :param index: the index of the point to return

        :return: specific point/s as numpy nX3 ndarray
        """
        return np.asarray(self.data.points)[index, :]

    def InitializeOpen3dObject(self, inputPoints):
        """
        Initializes the object according to the type of the input points

        :param inputPoints: the points from which the object should be initialized to

        :type inputPoints: np.ndarray, o3D.PointCloud, PointSet

        :return:
        """
        if isinstance(inputPoints, np.ndarray):
            self.data = O3D.PointCloud()
            self.data.points = O3D.Vector3dVector(inputPoints)
        elif isinstance(inputPoints, PointSet):
            self.data = O3D.PointCloud()
            pts = inputPoints.ToNumpy()[:, :3]
            self.data.points = O3D.Vector3dVector(pts)

        elif isinstance(inputPoints, O3D.PointCloud):
            self.data = inputPoints
        else:
            print("Given type: " + str(type(inputPoints)) + " as input. Not sure what to do with that...")
            raise ValueError("Wrong turn.")

    def RebuildKDTree(self, verbose=True):
        """
        Builds the KD-tree again

        :param verbose:
        :return:
        """
        if verbose:
            print("Rebuilding KDTree...")
        self.kdTreeOpen3D = O3D.KDTreeFlann(self.data)

    def ToNumpy(self):
        """
        Convert data to numpy array
        :return:
        """
        pointsArray = np.asarray(self.data.points)
        return pointsArray

    @property
    def Size(self):
        return len(self.data.points)


    def DownsampleCloud(self, voxelSize, verbose=True):
        if voxelSize > 0.:
            self.pointsOpen3D = O3D.voxel_down_sample(self.data, voxel_size=voxelSize)

            if verbose:
                print("Downsampling the point cloud with a voxel size of " + str(voxelSize))
                print("Number of points after down sampling: " + str(self.data))

            self.voxelSize = voxelSize
            self.numberOfPoints = len(self.data.points)
            # self.pointsNeighborsArray = np.empty(shape=(self.numberOfPoints,), dtype=PointNeighborhood)
            self.RebuildKDTree()

    def CalculateNormals(self, search_radius=0.05, maxNN=20, orientation=(0., 0., 0.), verbose=True):
        """
        Compute normals for PointSetOpen3D according to radius and maximum neighbors, if an orientation is given, the normals are computed towards the orientation.

        :param search_radius: neighbors radius for normal computation. Default: 0.05
        :param maxNN: maximum neighbors in a neighborhood. If set to (-1), there is no limitation. Default: 20.
        :param orientation: "camera" orientation. The orientation towards which the normals are computed. Default: (0,0,0)
        :param verbose: print inter-running messages.

        :type search_radius: float
        :type maxNN: int
        :type orientation: tuple
        :type verbose: bool

        :return:
        """

        print(">>> Calculating point-cloud normals. Neighborhood Parameters -- r:" + str(
            search_radius) + "\tnn:" + str(
                maxNN))

        if maxNN <= 0:
            O3D.estimate_normals(self.data,
                                 search_param=O3D.KDTreeSearchParamRadius(radius=search_radius))
        elif search_radius <= 0:
            O3D.estimate_normals(self.data,
                                 search_param=O3D.KDTreeSearchParamKNN(knn=maxNN))
        else:
            O3D.estimate_normals(self.data,
                                 search_param=O3D.KDTreeSearchParamHybrid(radius=search_radius, max_nn=maxNN))

        if isinstance(orientation, tuple):
            if orientation == (0., 0., 0.):
                O3D.orient_normals_towards_camera_location(self.data)  # Default Camera Location is (0, 0, 0).
            else:
                raise NotImplementedError("Need to modify...")
                O3D.orient_normals_to_align_with_direction(self.data)  # Default Direction is (0, 0, 1).
        else:
            raise ValueError("Orientation should be a tuple representing a location (X, Y, Z).\n"
                             "Default Location: Camera (0., 0., 0.).")

    def DisregardOtherPoints(self, indx):
        """
        **OBSOLETE** - should be removed

        Remove points from the PointSetOpen3D object

        :param indx: indices to remove

        :return:
        """
        from warnings import warn
        warn(DeprecationWarning)

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
        **OBSOLETE** - should be removed

        Remove points from the PointSetOpen3D object

        :param indx: indices to remove
        :param verbose: runtime printing. Default: True

        :type indx: list or int
        :type verbose: bool

        """
        from warnings import warn
        warn(DeprecationWarning)

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

    # region Visualization Functions
    def SetPointsColors(self, colors):
        """
        **OBSOLETE** -- should be removed

        :param colors:
        :return:
        """
        from warnings import warn
        warn(DeprecationWarning)

        if isinstance(colors, O3D.Vector3dVector):
            self.pointsOpen3D.colors = colors
        else:
            self.pointsOpen3D.colors = O3D.Vector3dVector(colors)

    def VisualizeClusters(self, pointsLabels, both=False):
        """
        **OBSOLETE** -- should be removed
        :param pointsLabels:
        :param both:
        :return:
        """
        from warnings import warn
        warn(DeprecationWarning)
        assert self.numberOfPoints == len(pointsLabels), "Dimensions do not match."

        clustersIDs = set(pointsLabels)
        numberOfClusters = len(clustersIDs)

        randomColors = LetThereBeRandomColors().GenerateNewColor(numberOfColorsToGenerate=numberOfClusters)

        for currentInd in range(1, numberOfClusters + 1):
            selectedColor = randomColors[currentInd - 1]
            np.asarray(self.pointsOpen3D.colors)[pointsLabels == currentInd] = selectedColor

        self.Visualize(both=both)

    def Visualize(self, original=False, both=False):
        """
        **OBSOLETE** -- is about to be removed

        :param original:
        :param both:
        :return:
        """
        import warnings

        warnings.warn(DeprecationWarning)

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
