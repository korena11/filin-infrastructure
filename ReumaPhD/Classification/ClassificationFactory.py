'''
infraGit
photo-lab-3\Reuma
21, Nov, 2017 
'''

# TODO: - create hypothesis functions for each classification (stopped at ridge)
# TODO: - make sure that idx do not repeat in multiple classes

# if platform.system() == 'Linux':

import numpy as np
from scipy.stats import norm

from EigenFactory import EigenFactory
from RasterData import RasterData
from ReumaPhD.Classification.ClassificationProperty import ClassificationProperty

# Classification codes
RIDGE = 1
PIT = 2
VALLEY = 3
FLAT = 4
PEAK = 5
SADDLE = 6


class ClassificationFactory:
    z_min = z_max = None
    oneTail = TwoTail = None
    percentMap = None
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
    def __checkIdx(self, a, b, new_percentMap):
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
                tmpidx = new_percentMap[repeated_Idx[:, 0], repeated_Idx[:, 1]] >= \
                         self.percentMap[repeated_Idx[:, 0], repeated_Idx[:, 1]]
                newIdx = repeated_Idx[tmpidx]
            else:
                for j in a:
                    if np.any((repeated_Idx[:] == j.tolist()).all(1)):
                        if new_percentMap[(j[0], j[1])] >= self.percentMap[(j[0], j[1])]:
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
    def __pit(self, oneTail, percentMap):
        """
         Hypothesis test for pit:
        :math: lambda_min > eighThreshold
        (and therefore lambda_max also)
        Reject when :math: z_min>z_1-alpha; Rejected are the pits

        :param oneTail: alpha for oneTail
        :param percentMap: the current percent map for classification
        :return:
        """

        new_percentMap = np.zeros(self.data.shape)
        idx_assigned = np.nonzero(self.classified.classified_map)

        idx = np.nonzero(self.z_min > oneTail)

        # Correct classification and percentMap
        new_percentMap[idx] = percentMap[idx]
        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_percentMap))
        if not np.any(newidx): return False

        self.percentMap[(newidx[:, 0], newidx[:, 1])] = new_percentMap[(newidx[:, 0], newidx[:, 1])]

        self.classified.classify_map(newidx, PIT)
        return newidx

    @classmethod
    def __peak(self, oneTail, percentMap):
        """
         Hypothesis test for peak:
         lambda_max < eighThreshold (and therefore lambda_min also)
        Reject when :math: z_max<-z_1-alpha; Rejected are the peaks

        :param oneTail:
        :param percentMap:
        :return:
        """
        new_percentMap = np.zeros(self.data.shape)

        idx_assigned = np.nonzero(self.classified.classified_map)

        idx = np.nonzero(self.z_max < -oneTail)
        new_percentMap[idx] = percentMap[idx]
        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_percentMap))
        if not np.any(newidx): return False

        self.percentMap[(newidx[:, 0], newidx[:, 1])] = new_percentMap[(newidx[:, 0], newidx[:, 1])]

        self.classified.classify_map(newidx, PEAK)
        return newidx

    @classmethod
    def __flat(self, twoTail, percentMap):
        """
        Hypothesis test for flat:
        lambda_min = eighThreshold and lambda_max = eigThreshold
        """
        new_percentMap = np.zeros(self.data.shape)

        idx_assigned = np.nonzero(self.classified.classified_map)

        idx = np.nonzero((np.abs(self.z_max) < twoTail) * (np.abs(self.z_min) < twoTail))
        new_percentMap[idx] = percentMap[idx]

        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_percentMap))
        if not np.any(newidx): return False

        self.percentMap[(newidx[:, 0], newidx[:, 1])] = new_percentMap[(newidx[:, 0], newidx[:, 1])]

        self.classified.classify_map(newidx, FLAT)
        return newidx

    @classmethod
    def __saddle(self, oneTail, percentMap1, percentMap2):
        """
        Hypothesis test for saddle:
        :math: lambda_min < eighThreshold and lambda_max > eigThreshold

        :param oneTail:
        :param percentMap1:
        :param percentMap2:
        :return:
        """

        new_percentMap = np.zeros(self.data.shape)

        idx_assigned = np.nonzero(self.classified.classified_map)

        idx = np.nonzero((self.z_min > oneTail) * (self.z_max < -oneTail))
        if not np.any(idx): return False
        new_percentMap[idx] = max(percentMap1[idx], percentMap2[idx])
        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_percentMap))
        if not np.any(newidx): return False

        self.percentMap[(newidx[:, 0], newidx[:, 1])] = new_percentMap[(newidx[:, 0], newidx[:, 1])]

        self.classified.classify_map(newidx, SADDLE)
        return newidx

    @classmethod
    def __ridge(self, oneTail, twoTail, percentMap):

        new_percentMap = np.zeros(self.data.shape)
        idx_assigned = np.nonzero(self.classified.classified_map)
        # Hypothesis test for ridge:
        #    1.    lambda_min = eighThreshold and lambda_max > eigThreshold
        # or 2.    lambda_max = eighThreshold and lambda_min > eigThreshold

        # 1. reject when  |lambda_min| > z_1-alpha/2 (choose non-rejected) AND when
        #                 lambda_max < -Z_1-alpha (choose reject)
        idx = np.nonzero((np.abs(self.z_min) < oneTail) * (self.z_max < -twoTail))
        new_percentMap[idx] = percentMap[idx]

        # 2. reject when |lambda_max| > z_1-alpha/2 (choose non-rejected) AND when
        #                lambda_min > Z_1-alpha (choose reject)

        idx2 = np.nonzero(np.abs(self.z_max) < twoTail * (self.z_min < -oneTail))
        new_percentMap[idx2] = (self.z_min[idx2] + self.z_max[idx2]) / 100.

        if idx[0].size != 0 and idx2[0].size != 0:
            idx = np.hstack((np.array(idx), np.array(idx2)))

        if idx[0].size == 0:
            idx = idx2
        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_percentMap))
        if not np.any(newidx): return False

        self.percentMap[(newidx[:, 0], newidx[:, 1])] = new_percentMap[(newidx[:, 0], newidx[:, 1])]

        self.classified.classify_map(newidx, RIDGE)
        return newidx

    @classmethod
    def __valley(self, oneTail, twoTail, percentMap):
        '''
         Hypothesis test for valley:
            1.    lambda_min = eighThreshold and lambda_max < eigThreshold
         or 2.    lambda_max = eighThreshold and lambda_min < eigThreshold
        '''
        new_percentMap = np.zeros(self.data.shape)
        idx_assigned = np.nonzero(self.classified.classified_map)
        # 1. reject when |lambda_min| > z_1-alpha/2 (choose non-rejected) AND when
        #                lambda_max > Z_1-alpha (choose reject)
        idx = np.nonzero((np.abs(self.z_min) < twoTail) * (self.z_max > oneTail))
        new_percentMap[idx] = percentMap[idx]

        # 2. reject when |lambda_max| > z_1-alpha/2 (choose non-rejected) AND when
        #                lambda_min < -Z_1-alpha (choose reject)

        idx2 = np.nonzero((np.abs(self.z_max) < twoTail) * (self.z_min > oneTail))
        new_percentMap[idx2] = (self.z_min[idx2] + self.z_max[idx2]) / 100.

        if idx[0].size != 0 and idx2[0].size != 0:
            idx = np.hstack((np.array(idx), np.array(idx2)))

        if idx[0].size == 0:
            idx = idx2

        newidx = np.array(self.__checkIdx(np.array(idx).T, np.array(idx_assigned).T, new_percentMap))
        if not np.any(newidx): return False

        self.percentMap[(newidx[:, 0], newidx[:, 1])] = new_percentMap[(newidx[:, 0], newidx[:, 1])]

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

        winsize = np.int(np.floor(winsize))
        if winsize % 2 == 0:
            winsize += 1

        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            alpha = 0.05

        if 'classProp' in kwargs:
            self.classified = kwargs['classProp']
        else:
            self.classified = ClassificationProperty(self.data)

        oneTail = norm.ppf(1 - alpha)
        twoTail = norm.ppf(1 - alpha / 2)

        if isinstance(self.data, RasterData):
            if self.percentMap is None:
                self.percentMap = np.zeros(self.data.shape)
            resolution = self.data.resolution
            eigenvalue_sigma = np.sqrt(6) * self.data.accuracy / (winsize * resolution) ** 2
            eigenprop = EigenFactory.eigen_Hessian(self.data, winsize=winsize, resolution=resolution)
            eigThreshold = 2 * self.data.roughness / (winsize * resolution) ** 2

            # Computes percentages for each combination:
            eigMax = eigenprop.eigenValues[1, :, :]
            eigMin = eigenprop.eigenValues[0, :, :]
            denominator_oneTail = oneTail * eigenvalue_sigma + eigThreshold
            denominator_twoTail = twoTail * eigenvalue_sigma

            eigMaxPos_eigMinPos = self.__percentMapComputation(eigenprop, 1, 1, denominator_oneTail)
            eigMaxPos_eigMinZero = self.__percentMapComputation(eigenprop, 1, 0,
                                                                denominator_oneTail, denominator_twoTail)
            eigMaxPos_eigMinNeg = self.__percentMapComputation(eigenprop, 1, -1, denominator_oneTail)

            eigMaxNeg_eigMinNeg = self.__percentMapComputation(eigenprop, -1, -1, denominator_oneTail)
            eigMaxNeg_eigMinPos = self.__percentMapComputation(eigenprop, -1, 1, denominator_oneTail)

            eigMaxZero_eigMinNeg = self.__percentMapComputation(eigenprop, 0, -1,
                                                                denominator_oneTail, denominator_twoTail)

            # Compute z-statistic for maximal eigenvalue and for minimal eigenvalue:
            # z_max/min = (lambda_max/min - eigThreshold) / eigenvalue_sigma
            self.z_max = (eigMax - eigThreshold) / eigenvalue_sigma
            self.z_min = (eigMin - eigThreshold) / eigenvalue_sigma

            self.__pit(oneTail, eigMaxPos_eigMinPos)
            self.__peak(oneTail, eigMaxNeg_eigMinNeg)
            self.__ridge(oneTail, twoTail, eigMaxZero_eigMinNeg)
            self.__valley(oneTail, twoTail, eigMaxPos_eigMinZero)
            self.__flat(twoTail, eigMaxPos_eigMinZero)

            self.__saddle(oneTail, eigMaxPos_eigMinNeg, eigMaxNeg_eigMinPos)

            return self.classified

    @staticmethod
    def __percentMapComputation(eigenProp, eigMax_sign, eigMin_sign,
                                denominator_oneTail=0, denominator_twoTail=0):
        """
        computes the percent map according to the probability to have a negative or positive min/max eigenvalues

        :param eigenProp: eigenvalues property, which have max and min eigen values assigned
        :param eigMax_sign: computing probability for: +1, -1, 0
        :param eigMin_sign: computing probability for: +1, -1, 0
        :param denominator1,2: one or two scalars, computed according to:
         - for one tail hypothesis: oneTail * eigenvalue_sigma + eigThreshold
         - for two tail hypothesis: twoTail * eigenvalue_sigma
        -- computed in advance (scalar)

        :type eigenProp: EigenProperty
        :type denominator_oneTail: float
        :type denominator_twoTail: float

        :return: the percent map for the required hypothesis
        """
        eigMax = eigenProp.eigenValues[1, :, :]
        eigMin = eigenProp.eigenValues[0, :, :]

        if eigMax_sign == 0:
            eigMax_normed = np.abs(eigMax) / denominator_twoTail
        else:
            eigMax_normed = eigMax_sign * eigMax / denominator_oneTail

        if eigMin_sign == 0:
            eigMin_normed = np.abs(eigMin) / denominator_twoTail
        else:
            eigMin_normed = eigMin_sign * eigMin / denominator_oneTail

        percentMap = (eigMax_normed + eigMin_normed) / 2

        percentMap[percentMap < 0] = 0
        return percentMap

    @classmethod
    def SurfaceClassification(self, data, winSizes, **kwargs):
        """
        Classifying the surface to areas of the same kind via different scale s
        :param data:
        :param winSizes: window sizes between which the classification is made.

        :type winSizes: ndarray nx1

        :return:
            classification property created based on different scales.

        """
        self.data = data
        classified = ClassificationProperty(data)
        for win in winSizes:
            print(('current window size %.4f' % win))
            if isinstance(self.data, RasterData):
                self.__ClassifyPoints(win, classProp=classified)
        return classified, self.percentMap


if __name__ == '__main__':
    from IOFactory import IOFactory
    import cProfile, pstats
    import matplotlib

    matplotlib.use('TkAgg')

    raster = IOFactory.rasterFromAscFile(r'D:\Documents\ownCloud\Data\minearl_try.txt')
    winSizes = np.linspace(0.1, 10, 5)
    cProfile.run("ClassificationFactory.SurfaceClassification(raster, winSizes)", "{}.profile".format(__file__))
    s = pstats.Stats("{}.profile".format(__file__))
    # classified, percentMap = ClassificationFactory.SurfaceClassification(raster, winSizes)
    s.strip_dirs()
    s.sort_stats("time").print_stats(10)
    # plt.imshow(classified.classified_map)
    # plt.show()
