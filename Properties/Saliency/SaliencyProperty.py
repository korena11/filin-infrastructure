from BaseProperty import BaseProperty
from PointSet import PointSet


class SaliencyProperty(BaseProperty):

    def __init__(self, points, saliencyValues):
        """

        :param points: the point cloud
        :param saliencyValues: value for each point

        :type points: PointSet
        """
        super(SaliencyProperty, self).__init__(points)
        self.__saliency = saliencyValues

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
