'''
infraGit
photo-lab-3\Reuma
21, Nov, 2017 
'''

# TODO: - create hypothesis functions for each classification (stopped at ridge)
# TODO: - make sure that idx do not repeat in multiple classes

# if platform.system() == 'Linux':
import matplotlib

matplotlib.use('TkAgg')

import numpy as np
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
        aa = a.copy()
        bb = b.copy()
        c = np.intersect1d(aa.view(dtype), bb.view(dtype))

        return c.view(a.dtype).reshape(-1, byaxis)

    @classmethod
    def __checkIdx(self, a, b, new_precentMap):
        """
        Checks that each index belongs to one classification
        :return:
        """
        if np.size(a) == 0:
            return False
        if np.size(b) == 0:
            return a
        newIdx = []
        repeated_Idx = self.intersect2d(a, b)
        if repeated_Idx.shape[0] != 0:
            if repeated_Idx.shape[0] == a.shape[0]:
                tmpidx = new_precentMap[repeated_Idx[:, 0], repeated_Idx[:, 1]] >= \
                         self.precentMap[repeated_Idx[:, 0], repeated_Idx[:, 1]]
                newIdx = repeated_Idx[tmpidx]
            else:
                for j in a:
                    if np.any((repeated_Idx[:] == j.tolist()).all(1)):
                        if new_precentMap[(j[0], j[1])] >= self.precentMap[(j[0], j[1])]:
                            newIdx.append(j)
                    else:
                        newIdx.append(j)
        else:
            newIdx = a
        if len(newIdx) == 0:
            return False
        else:
            return newIdx

    @classmethod
    def __pit(self, oneTail):

        new_precentMap = np.zeros(self.data.shape)

        idx_assigned = np.nonzero(self.classified.classified_map)

        # Hypothesis test for pit:
        # lambda_min > eighThreshold (and therefore lambda_max also)
        # Reject when z_min>z_1-alpha; Rejected are the pits
        idx = np.nonzero(self.z_min > oneTail)

        # Correct classification and precentMap
        new_precentMap[idx] = (self.z_min[idx] + self.z_max[idx]) / 100.
        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_precentMap))
        if not np.any(newidx): return False

        self.precentMap[(newidx[:, 0], newidx[:, 1])] = new_precentMap[(newidx[:, 0], newidx[:, 1])]

        self.classified.classify_map(newidx, PIT)
        return newidx

    @classmethod
    def __peak(self, oneTail):

        new_precentMap = np.zeros(self.data.shape)

        idx_assigned = np.nonzero(self.classified.classified_map)

        # Hypothesis test for peak:
        # lambda_max < eighThreshold (and therefore lambda_min also)
        # Reject when z_max<-z_1-alpha; Rejected are the peaks
        idx = np.nonzero(self.z_max < -oneTail)
        new_precentMap[idx] = self.z_max[idx] / 100.
        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_precentMap))
        if not np.any(newidx): return False

        self.precentMap[(newidx[:, 0], newidx[:, 1])] = new_precentMap[(newidx[:, 0], newidx[:, 1])]

        self.classified.classify_map(newidx, PEAK)
        return newidx

    @classmethod
    def __flat(self, twoTail):
        '''
        Hypothesis test for flat:
        lambda_min = eighThreshold and lambda_max = eigThreshold
        '''
        new_precentMap = np.zeros(self.data.shape)

        idx_assigned = np.nonzero(self.classified.classified_map)

        idx = np.nonzero((np.abs(self.z_max) < twoTail) * (np.abs(self.z_min) < twoTail))
        new_precentMap[idx] = (self.z_min[idx] + self.z_max[idx]) / 100.

        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_precentMap))
        if not np.any(newidx): return False

        self.precentMap[(newidx[:, 0], newidx[:, 1])] = new_precentMap[(newidx[:, 0], newidx[:, 1])]

        self.classified.classify_map(newidx, FLAT)
        return newidx

    @classmethod
    def __saddle(self, oneTail):
        # Hypothesis test for saddle:
        #    lambda_min < eighThreshold and lambda_max > eigThreshold
        new_precentMap = np.zeros(self.data.shape)

        idx_assigned = np.nonzero(self.classified.classified_map)

        idx = np.nonzero((self.z_min > oneTail) * (self.z_max < -oneTail))
        new_precentMap[idx] = (self.z_max[idx] + self.z_min[idx]) / 100.
        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_precentMap))
        if not np.any(newidx): return False


        self.precentMap[(newidx[:, 0], newidx[:, 1])] = new_precentMap[(newidx[:, 0], newidx[:, 1])]

        self.classified.classify_map(newidx, SADDLE)
        return newidx

    @classmethod
    def __ridge(self, oneTail, twoTail):

        new_precentMap = np.zeros(self.data.shape)
        idx_assigned = np.nonzero(self.classified.classified_map)
        # Hypothesis test for ridge:
        #    1.    lambda_min = eighThreshold and lambda_max > eigThreshold
        # or 2.    lambda_max = eighThreshold and lambda_min > eigThreshold

        # 1. reject when  |lambda_min| > z_1-alpha/2 (choose non-rejected) AND when
        #                 lambda_max < -Z_1-alpha (choose reject)
        idx = np.nonzero((np.abs(self.z_min) < oneTail) * (self.z_max < -twoTail))
        new_precentMap[idx] = (self.z_min[idx] + self.z_max[idx]) / 100.

        # 2. reject when |lambda_max| > z_1-alpha/2 (choose non-rejected) AND when
        #                lambda_min > Z_1-alpha (choose reject)

        idx2 = np.nonzero(np.abs(self.z_max) < twoTail * (self.z_min < -oneTail))
        new_precentMap[idx2] = (self.z_min[idx2] + self.z_max[idx2]) / 100.

        if idx[0].size != 0 and idx2[0].size != 0:
            idx = np.hstack((np.array(idx), np.array(idx2)))

        if idx[0].size == 0:
            idx = idx2
        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_precentMap))
        if not np.any(newidx): return False

        self.precentMap[(newidx[:, 0], newidx[:, 1])] = new_precentMap[(newidx[:, 0], newidx[:, 1])]

        self.classified.classify_map(newidx, RIDGE)
        return newidx

    @classmethod
    def __valley(self, oneTail, twoTail):
        '''
         Hypothesis test for valley:
            1.    lambda_min = eighThreshold and lambda_max < eigThreshold
         or 2.    lambda_max = eighThreshold and lambda_min < eigThreshold
        '''
        new_precentMap = np.zeros(self.data.shape)
        idx_assigned = np.nonzero(self.classified.classified_map)
        # 1. reject when |lambda_min| > z_1-alpha/2 (choose non-rejected) AND when
        #                lambda_max > Z_1-alpha (choose reject)
        idx = np.nonzero((np.abs(self.z_min) < twoTail) * (self.z_max > oneTail))
        new_precentMap[idx] = (self.z_min[idx] + self.z_max[idx]) / 100.

        # 2. reject when |lambda_max| > z_1-alpha/2 (choose non-rejected) AND when
        #                lambda_min < -Z_1-alpha (choose reject)

        idx2 = np.nonzero((np.abs(self.z_max) < twoTail) * (self.z_min > oneTail))
        new_precentMap[idx2] = (self.z_min[idx2] + self.z_max[idx2]) / 100.

        if idx[0].size != 0 and idx2[0].size != 0:
            idx = np.hstack((np.array(idx), np.array(idx2)))

        if idx[0].size == 0:
            idx = idx2

        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_precentMap))
        if not np.any(newidx): return False

        self.precentMap[(newidx[:, 0], newidx[:, 1])] = new_precentMap[(newidx[:, 0], newidx[:, 1])]

        self.classified.classify_map(newidx, VALLEY)
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
            if self.precentMap is None:
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
            self.__ridge(oneTail, twoTail)
            self.__valley(oneTail, twoTail)
            self.__flat(twoTail)
            self.__saddle(oneTail)

            return self.classified

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
            print ('current window size %.4f' % win)
            if isinstance(self.data, RasterData):
                self.__ClassifyPoints(win, classProp = classified)
        return classified, self.precentMap

if __name__ == '__main__':
    from IOFactory import IOFactory
    import cProfile, pstats

    raster = IOFactory.rasterFromAscFile(r'D:\Documents\ownCloud\Data\minearl_try.txt')
    winSizes = np.linspace(0.1, 10, 5)
    cProfile.run("ClassificationFactory.SurfaceClassification(raster, winSizes)", "{}.profile".format(__file__))
    s = pstats.Stats("{}.profile".format(__file__))
    # classified, precentMap = ClassificationFactory.SurfaceClassification(raster, winSizes)
    s.strip_dirs()
    s.sort_stats("time").print_stats(10)
    # plt.imshow(classified.classified_map)
    # plt.show()
