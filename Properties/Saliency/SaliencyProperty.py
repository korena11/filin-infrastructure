from BaseProperty import BaseProperty
from PointSet import PointSet


class SaliencyProperty(BaseProperty):

    def __init__(self, points, saliencyValues=None, **kwargs):
        """

        :param points: the point cloud
        :param saliencyValues: value for each point

        :type points: PointSet
        """
        super(SaliencyProperty, self).__init__(points)

        if saliencyValues is None:
            import numpy as np
            self.__saliency = np.empty(self.Size)
        else:
            self.__saliency = saliencyValues

    def __next__(self):
        self.current += 1
        try:
            return self.getPointSaliency(self.current - 1)
        except IndexError:
            self.current = 0
            raise StopIteration

    def getValues(self):
        """
        Saliency values for all the point cloud

        """
        return self.__saliency

    def getPointSaliency(self, idx):
        """
        Retrieve the saliency value of a specific point

        :param idx: the point index

        :return: saliency value

        :rtype: float

        """
        return self.__saliency[idx]

    def setPointSaliency(self, idx, values):
        """
        Sets a saliency values to specific points

        :param idx: a list or array of indices (can be only one) for which the saliency values refer
        :param values: the saliency values to assign

        :type idx: list, np.ndarray, int
        :type values: float

        """
        self.__saliency[idx] = values
