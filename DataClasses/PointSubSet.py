# Framework Imports
import VisualizationUtils
from DataClasses.PointSet import PointSet
import numpy as np


class PointSubSet(PointSet):
    """
    Holds a subset of a PointSet
    
    Provides the same interface as PointSet        
    """

    def __init__(self, points, indices, path=None, intensity=None, range_accuracy=0.002, angle_accuracy=0.012,
                 measurement_accuracy=0.002, **kwargs):

        if isinstance(points, PointSet):
            self.data = points

        else:
            super(PointSubSet, self).__init__(points, path=path, intensity=intensity, range_accuracy=range_accuracy,
                                              angle_accuracy=angle_accuracy, measurement_accuracy=measurement_accuracy)

        self.indices = indices.astype('int')

    @property
    def Size(self):
        """
        :return: number of points

        """
        from numpy import array
        return array(self.indices).shape[0]

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
        if isinstance(self.data, PointSet):
            intensity = self.data.Intensity
        else:
            intensity = self.__intensity
        if isinstance(intensity, np.ndarray):
            return intensity[self.indices]
        else:
            return np.asarray(intensity)[self.indices]

    @property
    def X(self):
        """
        :return: X coordinates

        :rtype: nx1 nd-array

        """
        return self.data.X[self.indices]

    @property
    def Y(self):
        """

        :return: Y coordinates

        :rtype: nx1 nd-array

        """
        return self.data.Y[self.indices]

    @property
    def Z(self):
        """
        :return: Z coordinates

        :rtype: nx1 nd-array

        """
        return self.data.Z[self.indices]

    def GetPoint(self, index):
        """
           Retrieve specific point(s) by index (when the index is according to the subset and not to the original set). The type of the array is float32, with a "name" for each column (easier sorting)

           :param index: the index of the point to return

           :return: specific point/s as numpy nX3 ndarray
        """
        return self.ToNumpy()[index, :]


    def ToNumpy(self):
        """
        Return the points as numpy nX3 ndarray
        """
        from numpy import ndarray, asarray


        if isinstance(self.data, PointSet):
            return self.data.ToNumpy()[self.indices, :]

        elif isinstance(self.data, ndarray):
            return asarray(self.data[self.indices, :])

        else: #assuming that else is
            return asarray(self.data.points)[self.indices, :]


    def ToPolyData(self):
        """
        :return: tvtk.PolyData of the current PointSet

        """

        numpy_points = self.ToNumpy()[self.indices, :]
        vtkPolydata = VisualizationUtils.MakeVTKPointsMesh(numpy_points)
        return vtkPolydata
