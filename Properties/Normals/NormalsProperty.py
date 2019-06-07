from Properties.BaseProperty import BaseProperty


class NormalsProperty(BaseProperty):
    __normals = None

    def __init__(self, points, normals=None):
        super(NormalsProperty, self).__init__(points)

        from numpy import empty
        self.__normals = empty((self.Size, 3))
        self.load(normals)

    def __next__(self):
        self.current += 1
        try:
            return self.getPointNormal(self.current - 1)
        except IndexError:
            self.current = 0
            raise StopIteration

    @property
    def Normals(self):
        """
        Return points' normals 
        """  
        return self.__normals

    def load(self, normals, **kwargs):
        """
        Sets normals into the NormalsProperty object

        :param normals: the computed normals
        :param kwargs:
        :return:
        """
        if normals is not None:
            self.__normals = normals


    def getValues(self):
        return self.__normals

    @property  
    def dX(self):
        """
        Return normals X coordinates 
        """  
        return self.__normals[:, 0]
    
    @property  
    def dY(self):
        """
        Return normals Y coordinates  
        """  
        return self.__normals[:, 1]
    
    @property  
    def dZ(self):
        """
        Return normals Z coordinates  
        """  
        return self.__normals[:, 2]

    def getPointNormal(self, idx):
        """
        Retrieve the normal value of a specific point

        :param idx: the point index

        :return: saliency value

         :rtype: np.ndarray
        """
        return self.__normals[idx, :]

    def setPointNormal(self, idx, values):
        """
        Sets a normal values to specific points

        :param idx: a list or array of indices (can be only one) for which the saliency values refer
        :param values: the saliency values to assign

        :type idx: list, np.ndarray, int
        :type values: np.ndarray (nx3)

        """
        self.__normals[idx, :] = values
