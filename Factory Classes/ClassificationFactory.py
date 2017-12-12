'''
infraGit
photo-lab-3\Reuma
21, Nov, 2017 
'''

# TODO: - create hypothesis functions for each classification (stopped at ridge)
# TODO: - make sure that idx do not repeat in multiple classes

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from RasterData import RasterData
from EigenFactory import EigenFactory
from ClassificationProperty import ClassificationProperty

# Classification codes
RIDGE  =    1
PIT    =    2
VALLEY =    3
FLAT   =    4
PEAK   =    5
SADDLE =    6


class ClassificationFactory:
    z_min = z_max = None
    oneTail = TwoTail = None
    precentMap = None
    data = None
    classified = None

    @staticmethod
    def intersect2d(a, b, **kwargs):
        """
        finds the recurring elements in two arrays (row-wise)

        :param axis: column-wise
        :return:
        """
        axis = kwargs.get('axis', 0)
        nrows, ncols = a.shape
        if axis == 0:
            byaxis = ncols

        else:
            # TODO: CHECK THIS ONE WROKS, DIDN'T TRY THAT
            byaxis = nrows
        dtype = {'names': ['f{}'.format(i) for i in range(byaxis)],
                 'formats': byaxis * [a.dtype]}

        c = np.intersect1d(a.view(dtype), b.view(dtype))

        return c.view(a.dtype).reshape(-1, byaxis)

    @classmethod
    def __checkIdx(self, a, b, new_precentMap):
        """
        Checks that each index belongs to one classification
        :return:
        """
        newIdx = []
        repeated_Idx = self.intersect2d(a, b)
        if repeated_Idx.shape[0] != 0:
            for j in a:
                if np.any(j == repeated_Idx):
                    for i in repeated_Idx:
                        if new_precentMap[i] >= self.precentMap[i]:
                            self.precentMap[i] = new_precentMap[i]
                            newIdx.append(i)
                else:
                    newIdx.append(j)
        else:
            newIdx = a
        return newIdx

    @classmethod
    def __pit(self, statistic):

        new_precentMap = np.zeros(self.data.shape)

        idx_assigned = np.nonzero(self.precentMap)

        # Hypothesis test for pit:
        # lambda_min > eighThreshold (and therefore lambda_max also)
        # Reject when z_min>z_1-alpha; Rejected are the pits
        idx = np.nonzero(self.z_min > statistic)

        # Correct classification and precentMap
        new_precentMap[idx] = self.z_min[idx] + self.z_max[idx]
        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_precentMap))

        self.classified.pit_idx = newidx
        return newidx

    @classmethod
    def __peak(self, statistic):

        new_precentMap = np.zeros(self.data.shape)

        idx_assigned = np.nonzero(self.precentMap)

        # Hypothesis test for peak:
        # lambda_max < eighThreshold (and therefore lambda_min also)
        # Reject when z_max<-z_1-alpha; Rejected are the peaks
        idx = np.nonzero(self.z_max < -statistic)

        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_precentMap))

        self.classified.peak_idx = newidx
        return newidx

    @classmethod
    def __ridge(self, statistic):

        new_precentMap = np.zeros(self.data.shape)

        idx_assigned = np.nonzero(self.precentMap)

        # Hypothesis test for ridge:
        #    1.    lambda_min = eighThreshold and lambda_max > eigThreshold
        # or 2.    lambda_max = eighThreshold and lambda_min > eigThreshold

        # 1. reject when  |lambda_min| > z_1-alpha/2 (choose non-rejected) AND when
        #                 lambda_max < -Z_1-alpha (choose reject)
        ridge_map = np.zeros(data.shape)
        ridge_map[(np.abs(z_min) < statistic) * (z_max < -statistic)] = RIDGE
        precentMap[(np.abs(z_min) < twoTail) * (z_max < -oneTail)] = z_min[(np.abs(z_min) < twoTail) * (
            z_max < -oneTail)] \
                                                                     + z_max[(np.abs(z_min) < twoTail) * (
            z_max < -oneTail)]

        # 2. reject when |lambda_max| > z_1-alpha/2 (choose non-rejected) AND when
        #                lambda_min < -Z_1-alpha (choose reject)
        ridge_map[] = RIDGE
        precentMap[(np.abs(z_max) < twoTail) * (z_min < -oneTail)] = z_min[(np.abs(z_max) < twoTail) * (
            z_min < -oneTail)] \
                                                                     + z_max[(np.abs(z_max) < twoTail) * (
            z_min < -oneTail)]

        idx = np.nonzero(np.abs(self.z_max) < self.twoTail) * (self.z_min < -twoTail)

        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_precentMap))

        self.classified.peak_idx = newidx
        return newidx

    @classmethod
    def __ClassifyPoints(self, winsize, **kwargs):
        """
        Classifying points according to their eigenvalues
        :param data: Raster or PointSet data
        :param winsize: the window size according to which teh classification is made
        :param classProp: a classification property that already exists .
        :param resolution: data resolution (cell size or scanning resolution)
        :param alpha: significance level for hypothesis testing. Default: 5%

                Default: no property
        :return: a classificaiton map
        """
        # TODO Adjust function for point clouds. Now works for raster only

        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            alpha = 0.05

        if 'classProp' in kwargs:
            self.classified = kwargs['classProp']
        else:
            self.classified = ClassificationProperty(self.data)

        oneTail = norm.ppf(1-alpha)
        twoTail = norm.ppf(1-alpha/2)

        if isinstance(self.data, RasterData):
            self.precentMap = np.zeros(self.data.shape)
            resolution = self.data.resolution
            eigenvalue_sigma = np.sqrt(6) * self.data.accuracy / (winsize * resolution) ** 2
            eigenprop = EigenFactory.eigen_Hessian(self.data, winsize = winsize, resolution = resolution)
            eigThreshold = 2 * self.data.roughness / (winsize * resolution) **2


            # Compute z-statistic for maximal eigenvalue and for minimal eigenvalue:
            # z_max/min = (lambda_max/min - eigThreshold) / eigenvalue_sigma
            self.z_max = (eigenprop.eigenValues[1, :, :] - eigThreshold) / eigenvalue_sigma
            self.z_min = (eigenprop.eigenValues[0, :, :] - eigThreshold) / eigenvalue_sigma

            self.__pit(oneTail)
            self.__peak(oneTail)






            # Hypothesis test for valley:
            #    1.    lambda_min = eighThreshold and lambda_max < eigThreshold
            # or 2.    lambda_max = eighThreshold and lambda_min < eigThreshold


            # 1. reject when |lambda_min| > z_1-alpha/2 (choose non-rejected) AND when
            #                lambda_max > Z_1-alpha (choose reject)
            valley_map = np.zeros(data.shape)
            valley_map[(np.abs(z_min) < twoTail) * (z_max > oneTail)] = VALLEY
            precentMap[(np.abs(z_min) < twoTail) * (z_max > oneTail)] = z_min[(np.abs(z_min) < twoTail) * (
            z_max > oneTail)] \
                                                                        + z_max[(np.abs(z_min) < twoTail) * (
            z_max > oneTail)]

            # 2. reject when |lambda_max| > z_1-alpha/2 (choose non-rejected) AND when
            #                lambda_min > Z_1-alpha (choose reject)
            valley_map[(np.abs(z_max) < twoTail) * (z_min > oneTail)] = VALLEY
            precentMap[(np.abs(z_max) < twoTail) * (z_min > oneTail)] = z_min[(np.abs(z_max) < twoTail) * (
            z_min > oneTail)] \
                                                                        + z_max[(np.abs(z_max) < twoTail) * (
            z_min > oneTail)]

            # Hypothesis test for flat:
            #    lambda_min = eighThreshold and lambda_max = eigThreshold
            flat_map = np.zeros(data.shape)
            flat_map[(np.abs(z_max)< twoTail) * (np.abs(z_min)< twoTail)] = FLAT
            precentMap[(np.abs(z_max) < twoTail) * (np.abs(z_min) < twoTail)] = z_min[(np.abs(z_max) < twoTail) * (
            np.abs(z_min) < twoTail)] \
                                                                                + z_max[(np.abs(z_max) < twoTail) * (
            np.abs(z_min) < twoTail)]

            # Hypothesis test for saddle:
            #    lambda_min < eighThreshold and lambda_max > eigThreshold
            saddle_map = np.zeros(data.shape)
            saddle_map[(z_min > oneTail) * (z_max <-oneTail)] = SADDLE
            precentMap[(z_min > oneTail) * (z_max < -oneTail)] = z_min[(z_min > oneTail) * (z_max < -oneTail)] \
                                                                 + z_max[(z_min > oneTail) * (z_max < -oneTail)]

            return ClassificationProperty(data, pit_map, valley_map, ridge_map, peak_map, flat_map,
                                          saddle_map), precentMap

    @classmethod
    def SurfaceClassification(self, data, winSizes, **kwargs):
        """
        Classifying the surface to areas of the same kind via different scale s
        :param data:
        :param winSizes: window sizes between which the classification is made. ndarray nx1
        :return:
            classification property created based on different scales.
        """
        self.data = data
        classified = ClassificationProperty(data)
        for win in winSizes:
            if isinstance(self.data, RasterData):
                self.__ClassifyPoints(win, classProp = classified)






if __name__ == '__main__':
    from IOFactory import IOFactory

    raster = IOFactory.rasterFromAscFile(r'D:\Documents\ownCloud\Data\sinkholei11.asc')
    winSizes = np.linspace(0.1, 10)
    classified, precentMap = ClassificationFactory.SurfaceClassification(raster, winSizes)
    plt.imshow(classified.ridge)
    plt.show()
