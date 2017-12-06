'''
infraGit
photo-lab-3\Reuma
21, Nov, 2017 
'''

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

import numpy as np
from BaseProperty import BaseProperty
from RasterData import RasterData
from PointSet import PointSet

# Classification codes
RIDGE  =    1
PIT    =    2
VALLEY =    3
FLAT   =    4
PEAK   =    5
SADDLE =    6


class ClassificationProperty(BaseProperty):
    classified_map = None

    def __init__(self, data, *args):
        """

        :param data: data according to which the property is built
        :param args:
        """
        super(ClassificationProperty, self).__init__(data)

        if isinstance(data, RasterData):
            self.classified_map = np.zeros(self.Raster.shape)
            self.classification(*args)

        elif isinstance(data, PointSet):
                #TODO
                pass

    def classification(self, *args):
        """
        Return classification map (numpy array)
        default: returns the entire classification map.

        Can be specific which:
        :param RIDGE
        :param PIT
        :param VALLEY
        :param FLAT
        :param PEAK
        :param SADDLE

        """
        classified_map = np.zeros(self.Raster.shape)
        if len(args) == 0:
            return self.classified_map

        else:
            for classification, map in enumerate(args):
                self.classified_map += map
            return self.classified_map

    @property
    def ridge(self):
        classisfied = np.zeros(self.classified_map.shape)
        classisfied[self.classified_map == RIDGE] = 1
        return classisfied \
 \
               @ property

    def pit(self):
        classisfied = np.zeros(self.classified_map.shape)
        classisfied[self.classified_map == PIT] = 1
        return classisfied

    @property
    def valley(self):
        classisfied = np.zeros(self.classified_map.shape)
        classisfied[self.classified_map == VALLEY] = 1
        return classisfied

    @property
    def peak(self):
        classisfied = np.zeros(self.classified_map.shape)
        classisfied[self.classified_map == PEAK] = 1
        return classisfied

    @property
    def flat(self):
        classisfied = np.zeros(self.classified_map.shape)
        classisfied[self.classified_map == FLAT] = 1
        return classisfied

    @property
    def saddle(self):
        classisfied = np.zeros(self.classified_map.shape)
        classisfied[self.classified_map == SADDLE] = 1
        return classisfied

    @property
    def unclassified(self):
        classisfied = np.zeros(self.classified_map.shape)
        classisfied[self.classified_map == 0] = 1
        return classisfied

if __name__ == '__main__':
    pass