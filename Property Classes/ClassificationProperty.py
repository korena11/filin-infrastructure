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
        if data is None:
            return

        super(ClassificationProperty, self).__init__(data)

        self.datatype = type(data)
        self.pit_idx = None
        self.valley_idx = None
        self.ridge_idx = None
        self.peak_idx = None
        self.saddle_idx = None
        self.flat_idx = None

        if self.datatype == RasterData:
            self.classified_map = np.zeros(self.Raster.shape)

        elif self.datatype == np.ndarray:
            self.classified_map = np.zeros(data.shape)


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


        for num, type in enumerate(args):
            if type == PIT:
                self.classified_map[self.pit_idx] = PIT
            if type == VALLEY:
                self.classified_map[self.valley_idx] = VALLEY
            if type == RIDGE:
                self.classified_map[self.ridge_idx] = RIDGE
            if type == FLAT:
                self.classified_map[self.flat_idx] = FLAT
            if type == PEAK:
                self.classified_map[self.peak_idx] = PEAK
            if type == SADDLE:
                self.classified_map[self.saddle_idx] = SADDLE

        return self.classified_map

    def classify_map(self, indices, type):
        """
        Classifies the map according to the indices sent and the type of the classification

        :param indices: tuple(array(nx1), array(nx1))
        :param type: CAPITAL LETTERS according the the classification definitions at the beginning of the file.
        :return: classified map
        """
        if self.datatype == RasterData:
            self.classified_map[indices[:, 0], indices[:, 1]] = type

        return self.classified_map
        # TODO: for point cloud

    @property
    def ridge(self):
        if self.datatype == RasterData:
            self.ridge_idx = np.nonzero(self.classified_map == RIDGE)
        return self.ridge_idx

    @property
    def pit(self):
        if self.datatype == RasterData:
            self.pit_idx = np.nonzero(self.classified_map == PIT)
        return self.pit_idx

    @property
    def valley(self):
        if self.datatype == RasterData:
            self.valley_idx = np.nonzero(self.classified_map == VALLEY)

        return self.valley_idx

    @property
    def peak(self):
        if self.datatype == RasterData:
            self.peak_idx = np.nonzero(self.classified_map == PEAK)

        return self.peak_idx

    @property
    def flat(self):
        if self.datatype == RasterData:
            self.flat_idx = np.nonzero(self.classified_map[self.classified_map == FLAT])

        return self.flat_idx
    @property
    def saddle(self):
        if self.datatype == RasterData:
            self.saddle_idx = np.nonzero(self.classified_map[self.classified_map == SADDLE])
        return self.saddle_idx

    def unclassified(self):
        unclassisfied = np.zeros(self.classified_map.shape)
        unclassisfied[self.classified_map == 0] = 1
        return unclassisfied




if __name__ == '__main__':
    pass