# Class PointSet hold a set of un-ordered 2D or 3D points.

from numpy import arange, array, vstack, hstack
from tvtk.api import tvtk


class PointSet(object):
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

        properties = {'rgb': None,
                      'intensity': None,
                      'accuracy': .002}
        properties.update(kwargs)

        if points.shape[1] == 3:  # 3D Data
            self.__xyz = points
        elif points.shape[1] == 2:  # 2D Data
            self.__xyz = None
            self.__xy = points

            # TODO: missing rgb is the size of the points.

        self.__rgb = properties['rgb']
        self.__intensity = properties['intensity']
        self.__measurement_accuracy = properties['accuracy']

    @property
    def Size(self):
        """
        Return number of points 
        """
        return self.__xyz.shape[0]

    @property
    def FieldsDimension(self):
        """
        Return number of columns (channels) 
        """
        if self.__intensity != None and self.__rgb != None:
            return 7
        elif self.__intensity == None and self.__rgb != None:
            return 6
        elif self.__intensity != None and self.__rgb == None:
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
        Return nX1 ndarray of X coordinate 
        """
        if self.__xyz != None:
            return self.__xyz[:, 0]
        else:
            return self.__xy[:, 0]

    @property
    def Y(self):
        """
        Return nX1 ndarray of Y coordinate 
        """
        if self.__xyz != None:
            return self.__xyz[:, 1]
        else:
            return self.__xy[:, 1]

    @property
    def Z(self):
        """
        Return nX1 ndarray of Z coordinate 
        """
        if self.__xyz != None:
            return self.__xyz[:, 2]
        else:
            return None

    @classmethod
    def ToNumpy(self):
        """
        Return the points as numpy nX3 ndarray (incase we change the type of __xyz in the future)
        """

        if self.__xyz != None:
            return self.__xyz
        elif self.__xy != None:
            return self.__xy

        return None

    @classmethod
    def ToPandas(cls):
        # TODO add this method
        pass

    @classmethod
    def GetPoint(self, index):
        """
        Return specific point/s as numpy nX3 ndarray (incase we change the type of __xyz in the future)
        """

        if self.__xyz != None:
            return self.__xyz[index, :]
        elif self.__xy != None:
            return self.__xy[index, :]

        return None

    @classmethod
    def UpdateFields(self, **kwargs):
        '''
        Update a field within the PointSet
        

        :param X, Y, Z: which field to update
        :param indices: which indices to update (optional)
        '''
        # TODO: add this option

        if 'X' in kwargs:
            self.__xyz[:, 0] = kwargs['X']

        if 'Y' in kwargs:
            self.__xyz[:, 1] = kwargs['Y']

        if 'Z' in kwargs:
            self.__xyz[:, 2] = kwargs['Z']

        if 'RGB' in kwargs:
            self.__rgb = kwargs['RGB']

        if 'XYZ' in kwargs:
            self.__xyz[:, :] = kwargs['XYZ']

        if 'XY' in kwargs:
            self.__xy[:, :] = kwargs['XY']

    def AddData2Fields(self, data, field = 'XYZ'):
        '''
        Add data to a field
        '''

        #TODO: add xy option
        if field == 'XYZ':
            self.__xyz = vstack((self.__xyz, data))
        if field == 'RGB':
            self.__rgb = vstack((self.__rgb, data))
        if field == 'Intensity':
            self.__intensity = hstack((self.__intensity, data))

    @classmethod
    def ToPolyData(self):
        """
        Create and return PolyData object
        
        :Return:
            - tvtk.PolyData of the current PointSet
        """
        # TODO add xy option?

        _polyData = tvtk.PolyData(points = array(self.__xyz, 'f'))
        verts = arange(0, self.__xyz.shape[0], 1)
        verts.shape = (self.__xyz.shape[0], 1)
        _polyData.verts = verts

        return _polyData
