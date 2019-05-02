# Utils Imports
# General Imports
import numpy as np
import open3d as O3D

from PointSet import PointSet

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
            self.path = inputPoints.path

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
            search_radius) + "\tnn:" + str(maxNN))

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
