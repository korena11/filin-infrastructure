import numpy as np
import open3d as O3D

# Framework Imports
from PointSetOpen3D import PointSetOpen3D


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

    def Visualize(self, original=False, both=False):
        # TODO: Elia please redo - so it will show only the subset

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
            originalColors = np.zeros((self.Size, 3), dtype=np.float)
            originalColors[self.indices, 0] = 0.5
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
