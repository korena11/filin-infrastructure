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
            self.pointSet = points
        else:
            self.pointSet = super(PointSubSet, self).__init__(points)

        self.indices = indices

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
    def RGB(self):
        """
        Return nX3 ndarray of rgb values    
        """
        rgb = self.pointSet.RGB
        if rgb == None:
            return None
        else:
            return self.pointSet.RGB[self.indices, :]

    @property
    def Intensity(self):
        """
        Return nX1 ndarray of intensity values 
        """
        intensity = self.pointSet.Intensity
        if intensity == None:
            return None
        else:
            return self.pointSet.Intensity[self.indices, :]

    @property
    def X(self):
        """
        Return nX1 ndarray of X coordinate 
        """
        return self.pointSet.X[self.indices]

    @property
    def Y(self):
        """
        Return nX1 ndarray of Y coordinate 
        """
        return self.pointSet.Y[self.indices]

    @property
    def Z(self):
        """
        Return nX1 ndarray of Z coordinate 
        """
        return self.pointSet.Z[self.indices]

    def ToNumpy(self):
        """
        Return the points as numpy nX3 ndarray (incase we change the type of __xyz in the future)
        
        :Return:
        
        """
        return self.pointSet.ToNumpy()[self.indices, :]

    def ToPolyData(self):
        """
        :Returns:
            tvtk.PolyData of the current PointSet
        """

        numpy_points = self.pointSet.ToNumpy()[self.indices, :]
        vtkPolydata = VisualizationUtils.MakeVTKPointsMesh(numpy_points)
        return vtkPolydata
