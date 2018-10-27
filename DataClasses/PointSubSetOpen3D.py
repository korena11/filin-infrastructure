import numpy as np

# Framework Imports
from PointSetOpen3D import PointSetOpen3D
from PointSubSet import PointSubSet


class PointSubSetOpen3D(PointSetOpen3D, PointSubSet):
    """
    Holds a subset of a PointSetOpen3D

    Provides the same interface as PointSetOpen3D and PointSubSet
    """

    def __init__(self, points, indices):

        if isinstance(points, PointSetOpen3D):
            self.pointSet = points
        else:
            self.pointSet = super(PointSubSetOpen3D, self).__init__(points)

        self.indices = indices

    def ToNumpy(self):
        """
        Return the points as numpy nX3 ndarray (incase we change the type of __xyz in the future)
        """

        pointsArray = np.asarray(self.originalPointsOpen3D.points)[self.indices, :]
        return pointsArray
