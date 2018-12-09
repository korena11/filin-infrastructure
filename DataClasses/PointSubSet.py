# Framework Imports
import VisualizationUtils
from PointSet import PointSet


class PointSubSet(PointSet):
    """
    Holds a subset of a PointSet
    
    Provides the same interface as PointSet        
    """

    def __init__(self, points, indices):

        if isinstance(points, PointSet):
            self.data = points
        else:
            super(PointSubSet, self).__init__(points)

        self.indices = indices

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
        intensity = self.Intensity
        if isinstance(intensity, np.ndarray):
            return self.Intensity[self.indices]
        else:
            return np.asarray(self.Intensity)[self.indices]

    def ToNumpy(self):
        """
        Return the points as numpy nX3 ndarray (incase we change the type of __xyz in the future)
        

        """
        from numpy import array
        return array(self.data)[self.indices, :]

    def ToPolyData(self):
        """
        :return: tvtk.PolyData of the current PointSet

        """

        numpy_points = self.ToNumpy()[self.indices, :]
        vtkPolydata = VisualizationUtils.MakeVTKPointsMesh(numpy_points)
        return vtkPolydata
