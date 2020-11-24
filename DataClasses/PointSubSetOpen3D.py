import numpy as np


# Framework Imports
from DataClasses.PointSetOpen3D import PointSetOpen3D


class PointSubSetOpen3D(PointSetOpen3D):
    """
    Holds a subset of a PointSetOpen3D

    Provides the same interface as PointSetOpen3D and PointSubSet
    """

    def __init__(self, points, indices):

        if isinstance(points, PointSetOpen3D):
            self.data = points.data

        else:
            super(PointSubSetOpen3D, self).__init__(points)

        self.indices = indices

    def ToNumpy(self):
        """
        Return the points as numpy nX3 ndarray (in case we change the type of __xyz in the future)
        """

        pointsArray = np.asarray(self.data.points)[self.indices, :]
        return pointsArray

    @property
    def Size(self):
        """
        Return number of points
        """
        return len(self.indices)

    @property
    def GetIndices(self):
        """
        Return points' indices
        """
        return self.indices

    @property
    def Intensity(self):
        """
        Return nX1 ndarray of intensity values
        """
        import numpy as np
        intensity = self.data.Intensity
        if isinstance(intensity, np.ndarray):
            return self.data.Intensity[self.indices]
        else:
            return None

    @property
    def X(self):
        """
        Return nX1 ndarray of X coordinate
        """

        return np.asarray(self.data.points)[self.GetIndices, 0]

    @property
    def Y(self):
        """
        Return nX1 ndarray of Y coordinate
        """
        return np.asarray(self.data.points)[self.GetIndices, 1]

    @property
    def Z(self):
        """
        Return nX1 ndarray of Z coordinate
        """
        return np.asarray(self.data.points)[self.GetIndices, 2]

