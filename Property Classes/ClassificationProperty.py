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
from matplotlib import pyplot as plt
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

    def __init__(self, dataset):
        super(ClassificationProperty, self).__init__(datasest = dataset)

        if isinstance(dataset, RasterData):
            self.classified_map = np.zeros(self.Raster.shape)

        elif isinstance(dataset, PointSet):
                #TODO
                pass

    # def classification(self, *args):
    #     """
    #     Return classification map (numpy array)
    #     default: returns the entire classification map.
    #
    #     Can be specific which:
    #     :param RIDGE
    #     :param PIT
    #     :param VALLEY
    #     :param FLAT
    #     :param PEAK
    #     :param SADDLE
    #
    #     """
    #     classified_map = np.zeros(self.Raster.shape)
    #     if len(args) == 0:
    #         return self.classified_map
    #
    #     else:
    #         for count, classification in enumerate(args):
    #
    #             classified_map += self.classified_map[self.classified_map == classification]
    #         return classified_map


if __name__ == '__main__':
    pass