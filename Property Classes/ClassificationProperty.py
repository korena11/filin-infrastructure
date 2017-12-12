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
        self.datatype = type(data)
        self.pit_idx = None
        self.valley_idx = None
        self.ridge_idx = None
        self.peak_idx = None
        self.saddle_idx = None
        self.flat_idx = None

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

        # TODO: append for PointSet data

        if self.datatype == RasterData:
            classified_map = np.zeros(self.Raster.shape)
            raster_flag = True
        for num, type in enumerate(args):
            if type == PIT:
                classified_map[self.pit_idx] = PIT
            if type == VALLEY:
                classified_map[self.valley_idx] = VALLEY
            if type == RIDGE:
                classified_map[self.ridge_idx] = RIDGE
            if type == FLAT:
                classified_map[self.flat_idx] = FLAT
            if type == PEAK:
                classified_map[self.peak_idx] = PEAK
            if type == SADDLE:
                classified_map[self.saddle_idx] = SADDLE

        return classified_map

    @property
    def ridge(self, idx = None):
        if idx is not None:
            self.ridge_idx = idx
        else:
            return self.ridge_idx

    @property
    def pit(self, idx = None):
        if idx is not None:
            self.pit_idx = idx
        else:
            return self.pit_idx

    @property
    def valley(self, idx = None):
        if idx is not None:
            self.valley_idx = idx
        else:
            return self.valley_idx

    @property
    def peak(self, idx = None):
        if idx is not None:
            self.peak_idx = idx
        else:
            return self.peak_idx

    @property
    def flat(self, idx = None):
        if idx is not None:
            self.flat_idx = idx
        else:
            return self.flat_idx

    @property
    def saddle(self, idx = None):
        if idx is not None:
            self.saddle_idx = idx
        else:
            return self.saddle_idx

    @property
    def unclassified(self):
        classisfied = np.zeros(self.classified_map.shape)
        classisfied[self.classified_map == 0] = 1
        return classisfied


if __name__ == '__main__':
    pass