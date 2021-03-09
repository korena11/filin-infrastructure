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

    def dip_direction(self):
        """
        Compute the dip direction

        The dip gives the steepest angle of descent of a tilted bed or feature relative to a horizontal plane,
        and is given by the number (0°-90°) as well as a letter (N,S,E,W) with rough direction in which the bed is dipping downwards.
        The dip direction is the azimuth of the direction the dip as projected to the horizontal (like the trend of a linear
        feature in trend and plunge measurements), which is 90° off the strike angle.

        The dip is computed by

        .. math::
            \theta = \arcsin \left(\frac{\sqrt{n_x^2 + n_y^2}}{\sqrt{n_x^2 + n_y^2+n_z^2}}\right)

        :return: the dip direction (decimal angels)
        :rtype: np.array nx1
        """
        import numpy as np

        numerator = np.sqrt(self.dX**2 + self.dY**2)
        denominator = np.sqrt(self.dX**2 + self.dY**2 + self.dZ**2)
        denominator[np.where(denominator == 0)] =1 # assuming dx and dy are zeros, the values where the denominator is zero will be zero
        return np.arcsin(numerator / denominator+1e-16)


