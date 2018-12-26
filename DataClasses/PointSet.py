# Class PointSet hold a set of un-ordered 2D or 3D points.

import numpy as np
from numpy import vstack, hstack

import VisualizationUtils
from BaseData import BaseData


# from vtk.api import vtk


class PointSet(BaseData):
    r"""
    Basic point cloud 
    
    Mandatory Data (must be different from None):
                
        __xyz (nX3 ndarray, n-number of points) - xyz unstructured Data (only 3D currently) 
        
    Optional Data (can be None or default value):
        - intensity - intensity of each point (ndarray)
        - range_accuracy: range measurement accuracy (default 0.002 m)
        - angle_accuracy: angle measurement accuracy (default  :math:`0.012^\circ`)
        - measurement_accuracy: noise of modeled surface (default 0.002 m)

         
    """

    def __init__(self, points, path=None, intensity=None, range_accuracy=0.002, angle_accuracy=0.012,
                 measurement_accuracy=0.002):
        r"""
        Initialize the PointSet object

        :param points: ndarray of xyz or xy

        **Optionals**

        :param intensity: intensity values for each point(optional)
        :param path: path to PointSet file
        :param range_accuracy: range measurement accuracy in m (default 0.002 m)
        :param angle_accuracy: angle measurement accuracy in degrees (default  :math:`0.012^\circ`)
        :param measurement_accuracy: noise of modeled surface (default 0.002 m)

        :type points: np.array, open3d.PointCloud, or any other thing...
        :type intensity: int
        :type range_accuracy: float
        :type angle_accuracy: float
        :type path: str

        """
        super(PointSet, self).__init__()

        self.data = points

        self.__intensity = intensity
        self.__range_accuracy = range_accuracy
        self.__angle_accuracy = angle_accuracy
        self.__measurement_accuracy = measurement_accuracy
        self.setPath(path)

        self.__current = 0  # for iterable object

    # ---------- Definitions to make iterable -----------
    def __iter__(self):
        return self

    def __next__(self):
        self.__current += 1
        try:
            return self.GetPoint(self.__current - 1)
        except IndexError:
            self.__current = 0
            raise StopIteration

    def __reset__(self):
        """
        Reset iterable
        :return:
        """
        self.__current = 0

        # --------end definitions for iterable object-----------

    @property
    def Size(self):
        """
        :return: number of points

        """
        return np.array(self.data).shape[0]

    @property
    def FieldsDimension(self):
        """
        Return number of columns (channels)
        """
        if self.__intensity is not None:
            return 4

        else:
            return 3

    @property
    def Intensity(self):
        """
        Return nX1 ndarray of intensity values 
        """
        return self.__intensity

    @property
    def X(self):
        """
        :return: X coordinates

        :rtype: nx1 nd-array

        """
        return self.ToNumpy()[:, 0]

    @property
    def Y(self):
        """

        :return: Y coordinates

        :rtype: nx1 nd-array

        """
        return self.ToNumpy()[:, 1]

    @property
    def Z(self):
        """
        :return: Z coordinates

        :rtype: nx1 nd-array

        """
        return self.ToNumpy()[:, 2]

    def ToNumpy(self):
        """
        :return: points as numpy nX3 ndarray
        """

        return np.array(self.data)

    def ToRDD(self):
        """
        Convert PointSet into pySpark Resilient Destributed Dataset (RDD)

        """
        import pyspark
        return pyspark.SparkContext.parallelize([self.X, self.Y, self.Z])

    #
    def ToGeoPandas(self, crs=None):
        """
        :param crs: coordinate spatial reference, if exists

        :return: pointSet as GeoPandas (geoseries) object (Points)
        :rtype: geopandas.geoseries


        #TODO: there might be a smarter way to do it

        """
        from pandas import DataFrame
        from geopandas import GeoDataFrame
        from shapely.geometry import Point

        # Transform to pandas DataFrame
        pts = DataFrame(self.ToNumpy())

        # Transform to geopandas GeoDataFrame
        geometry = [Point(xyz) for xyz in zip(self.X, self.Y, self.Z)]
        geodf = GeoDataFrame(pts, crs=crs, geometry=geometry)
        return geodf

    def GetPoint(self, index):
        """
        Retrieve specific point(s) by index

        :param index: the index of the point to return

        :return: specific point/s as numpy nX3 ndarray

        """
        return self.data[index, :]

    def UpdateFields(self, **kwargs):
        '''
        Update a field within the PointSet
        
        :param X, Y, Z: which field to update
        :param indices: which indices to update (optional)

        '''
        # TODO: add this option

        if 'X' in kwargs:
            self.data[:, 0] = kwargs['X']

        if 'Y' in kwargs:
            self.data[:, 1] = kwargs['Y']

        if 'Z' in kwargs:
            self.data[:, 2] = kwargs['Z']

        if 'XYZ' in kwargs:
            self.data[:, :] = kwargs['XYZ']

    def AddData2Fields(self, data, field='XYZ'):
        '''
        Add data to a field
        '''

        if field == 'XYZ' or field == 'xyz':
            self.data(vstack((self.data, data)))

        if field == 'Intensity' or field == 'intensity':
            if self.__intensity is None:
                self.__intensity = data
            else:
                self.__intensity = hstack((self.__intensity, data))

    def ToPolyData(self):
        """
        Create and return vtkPolyData object

        :return vtk.vtkPolyData of the current PointSet

        """
        numpy_points = self.ToNumpy()
        vtkPolyData = VisualizationUtils.MakeVTKPointsMesh(numpy_points)

        return vtkPolyData
