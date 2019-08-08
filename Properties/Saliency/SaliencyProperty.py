from Properties.BaseProperty import BaseProperty
from DataClasses.PointSet import PointSet


class SaliencyProperty(BaseProperty):

    def __init__(self, points, saliencyValues=None):
        """

        :param points: the point cloud
        :param saliencyValues: value for each point

        :type points: PointSet
        """
        import numpy as np

        super(SaliencyProperty, self).__init__(points)
        self.__normalize = False

        if saliencyValues is None:
            self.__saliency = np.empty(self.Size)
        else:
            self.__saliency = np.asarray(saliencyValues)

    def __next__(self):
        self.current += 1
        try:
            return self.getPointSaliency(self.current - 1)
        except IndexError:
            self.current = 0
            raise StopIteration

    def normalize_values(self, bool):
        """
        A flag whether to normalize values of principal curvature (k1, k2) to [0,1] with 2 being the invalid value

        :param bool: boolean flag for normalization of the values

        :type bool: bool

        """
        self.__normalize = bool

    def getValues(self):
        """
        Saliency values for all the point cloud

        .. warning::
            If normalized is True, the normalization is with log

        """
        saliency = self.__saliency

        if self.__normalize:
            import numpy as np
            # set values larger or smaller than 3sigma the average to invalid value
            s_tmp = self.__saliency.copy()
            s_tmp[np.where(saliency < saliency.mean() - saliency.std() * 3)] = saliency.mean() - 3 * saliency.std()
            s_tmp[np.where(saliency > saliency.mean() + saliency.std() * 3)] = saliency.mean() + 3 * saliency.std()

            saliency = np.log(s_tmp)

        return saliency

    def getPointSaliency(self, idx=None):
        """
        Retrieve the saliency value of a specific point

        :param idx: the point index

        :return: saliency value

        :rtype: float

        """
        if idx is None:
            return self.__saliency
        else:
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
