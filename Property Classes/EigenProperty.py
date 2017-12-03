'''
infraGit
photo-lab-3\Reuma
23, Nov, 2017
'''

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

from RasterData import RasterData
from PointSet import PointSet
from BaseProperty import BaseProperty


class EigenPropoerty(BaseProperty):

    __eigenVectors = None
    __eigenValues = None

    def __init__(self, dataset, **kwargs):
        """
        Property that holds eigen propoerties (e.g., eigen vectors, eigen values)
        :param dataset: a data class (PointSet of RasterData)
        :param kwargs:
                eigenValues: if two nd-arrays were sent for raster data: np.array([minVals, maxVals])
                eigenVectors
        """
        super(EigenPropoerty, self).__init__(dataset)

        if 'eigenValues' in kwargs:
            self.__eigenValues = kwargs['eigenValues']
        if 'eigenVectors' in kwargs:
            self.__eigenVectors = kwargs['eigenVectors']

    @property
    def eigenValues(self):
        return self.__eigenValues

    @property
    def eigenVectors(self):
        return self.__eigenVectors



if __name__ == '__main__':
    pass