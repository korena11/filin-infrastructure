# Class PointSet hold a set of un-ordered 2D or 3D points.

import numpy as np
from numpy import arange, array, vstack, hstack
from tvtk.api import tvtk

from BaseData import BaseData


class PointSet(BaseData):
    """ 
    Basic point cloud 
    
    Mandatory Data (must be different from None):
                
        __xyz (nX3 ndarray, n-number of points) - xyz unstructured Data (only 3D currently) 
        
    Optional Data (May not exist at all, or can be None):
        __rgb - color of each point (ndarray) 
        __intensity - intensity of each point (ndarray)
        __measurement_accuracy: noise of modeled surface
         
    """

    def __init__(self, points, **kwargs):
        """
        Initialize the PointSet object

        :param 'points': ndarray of xyz or xy
        :param 'rgb': rgb values for each point (optional)
        :param 'intensity': intensity values for each point(optional)
        """
        super(PointSet, self).__init__()
        properties = {'rgb': None,
                      'intensity': None,
                      'accuracy': .002}
        properties.update(kwargs)

        self.setdata(points)

        # if points.shape[1] == 3:  # 3D Data

        # elif points.shape[1] == 2:  # 2D Data

            # TODO: missing rgb is the size of the points.
        self.__rgb = properties['rgb']
        self.__intensity = properties['intensity']
        self.__measurement_accuracy = properties['accuracy']

        path = kwargs.get('path', '')  # The path for the data file
        self.setPath(path)

    @property
    def Size(self):
        """
        :return: number of points

        """
        return self.data.shape[0]

    @property
    def FieldsDimension(self):
        """
        Return number of columns (channels) 
        """
        if self.__intensity is not None and self.__rgb is not None:
            return 7
        elif self.__intensity is None and self.__rgb is not None:
            return 6
        elif self.__intensity is not None and self.__rgb is None:
            return 4
        else:
            return 3

    @property
    def RGB(self):
        """
        Return nX3 ndarray of rgb values 
        """
        return self.__rgb

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
        return self.data[:, 0]

    @property
    def Y(self):
        """

        :return: Y coordinates

        :rtype: nx1 nd-array

        """
        return self.data[:, 1]

    @property
    def Z(self):
        """
        :return: Z coordinates

        :rtype: nx1 nd-array

        """
        return self.data[:, 2]


    def ToNumpy(self):
        """
        :return: points as numpy nX3 ndarray
        """

        return np.array(self.data)


    @classmethod
    def ToPandas(cls):
        # TODO add this method
        pass

    def GetPoint(self, index):
        """
        :param index: the index of the point to return

        :return: pecific point/s as numpy nX3 ndarray

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

        if 'RGB' in kwargs:
            self.__rgb = kwargs['RGB']

        if 'XYZ' in kwargs:
            self.data[:, :] = kwargs['XYZ']


    def AddData2Fields(self, data, field = 'XYZ'):
        '''
        Add data to a field
        '''

        if field == 'XYZ':
            self.setdata(vstack((self.data, data)))
        if field == 'RGB':
            # TODO: check that this works
            self.__rgb = vstack((self.__rgb, data))

        if field == 'Intensity':
            self.__intensity = hstack((self.__intensity, data))

    def ToPolyData(self):
        """
        Create and return PolyData object

        :return tvtk.PolyData of the current PointSet

        """

        _polyData = tvtk.PolyData(points = array(self.data, 'f'))
        verts = arange(0, self.data.shape[0], 1)
        verts.shape = (self.data.shape[0], 1)
        _polyData.verts = verts

        return _polyData

