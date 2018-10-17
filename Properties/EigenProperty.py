'''
infraGit
photo-lab-3\Reuma
23, Nov, 2017
'''

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

from BaseProperty import BaseProperty


class EigenProperty(BaseProperty):
    """
    Property that holds eigen propoerties (e.g., eigen vectors, eigen values)

    """
    __eigenVectors = None
    __eigenValues = None

    def __init__(self, dataset, **kwargs):
        """
        Property that holds eigen propoerties (e.g., eigen vectors, eigen values)

        :param dataset: a data class (PointSet of RasterData)
        :param eigenValues: if two nd-arrays were sent for raster data: np.array([minVals, maxVals])
        :param eigenVectors

        """
        super(EigenProperty, self).__init__(dataset)
        self.setValues(**kwargs)

    def setValues(self, **kwargs):
        """
        Sets eigen values and eigen vectors into the EigenProperty object

        :param eigenValues: if two nd-arrays were sent for raster data. np.array([minVals, maxVals])
        :param eigenVectors:

        :type eigenValues: np.array
        :type eigenVectors: np.array

        """
        if 'eigenValues' in kwargs:
            self.__eigenValues = kwargs['eigenValues']
        if 'eigenVectors' in kwargs:
            self.__eigenVectors = kwargs['eigenVectors']

    def getValues(self):
        return self.__eigenValues

    @property
    def eigenValues(self):
        return self.__eigenValues

    @property
    def eigenVectors(self):
        return self.__eigenVectors


if __name__ == '__main__':
    pass