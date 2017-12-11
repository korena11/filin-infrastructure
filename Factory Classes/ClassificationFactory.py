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

    @staticmethod
    def __ClassifyPoints(data, winsize, **kwargs):
        """
        Classifying points according to their eigenvalues
        :param data: Raster or PointSet data
        :param winsize: the window size according to which teh classification is made
        :param resolution: data resolution (cell size or scanning resolution)
        :param significance level for hypothesis testing. Default: 5%
        :return: a classificaiton map
        """
        # TODO Adjust function for point clouds. Now works for raster only

        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            alpha = 0.05

        oneTail = norm.ppf(1-alpha)
        twoTail = norm.ppf(1-alpha/2)


        if isinstance(data, RasterData):
            precentMap = np.zeros(data.shape)
            resolution = data.resolution
            eigenvalue_sigma = np.sqrt(6) * data.accuracy / (winsize * resolution) ** 2
            eigenprop = EigenFactory.eigen_Hessian(data, winsize = winsize, resolution = resolution)
            eigThreshold = 2 * data.roughness / (winsize * resolution)**2


            # Compute z-statistic for maximal eigenvalue and for minimal eigenvalue:
            # z_max/min = (lambda_max/min - eigThreshold) / eigenvalue_sigma
            z_max = (eigenprop.eigenValues[1,:,:] - eigThreshold) / eigenvalue_sigma
            z_min = (eigenprop.eigenValues[0,:,:] - eigThreshold) / eigenvalue_sigma

            # Hypothesis test for pit:
            # lambda_min > eighThreshold (and therefore lambda_max also)
            # Reject when z_min>z_1-alpha; Rejected are the pits
            pit_map = np.zeros(data.shape)
            pit_map[z_min > oneTail] = PIT
            precentMap[z_min > oneTail] = z_min[z_min > oneTail] + z_max[z_max > oneTail]

            # Hypothesis test for peak:
            # lambda_max < eighThreshold (and therefore lambda_min also)
            # Reject when z_max<-z_1-alpha; Rejected are the peaks
            peak_map = np.zeros(data.shape)
            peak_map[z_max < -oneTail] = PEAK
            precentMap[z_min < -oneTail] = z_min[z_min < -oneTail] + z_max[z_max < -oneTail]

            # Hypothesis test for ridge:
            #    1.    lambda_min = eighThreshold and lambda_max > eigThreshold
            # or 2.    lambda_max = eighThreshold and lambda_min > eigThreshold

            # 1. reject when  |lambda_min| > z_1-alpha/2 (choose non-rejected) AND when
            #                 lambda_max < -Z_1-alpha (choose reject)
            ridge_map = np.zeros(data.shape)
            ridge_map[(np.abs(z_min) < twoTail) * (z_max < -oneTail)] = RIDGE
            precentMap[(np.abs(z_min) < twoTail) * (z_max < -oneTail)] = z_min[(np.abs(z_min) < twoTail) * (
            z_max < -oneTail)] \
                                                                         + z_max[(np.abs(z_min) < twoTail) * (
            z_max < -oneTail)]

            # 2. reject when |lambda_max| > z_1-alpha/2 (choose non-rejected) AND when
            #                lambda_min < -Z_1-alpha (choose reject)
            ridge_map[(np.abs(z_max) < twoTail) * (z_min < -oneTail)] = RIDGE
            precentMap[(np.abs(z_max) < twoTail) * (z_min < -oneTail)] = z_min[(np.abs(z_max) < twoTail) * (
            z_min < -oneTail)] \
                                                                         + z_max[(np.abs(z_max) < twoTail) * (
            z_min < -oneTail)]

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

    @staticmethod
    def SurfaceClassification(data, min_win_size, max_win_size):
        """
        Classifying the surface to areas of the same kind via different scale s
        :param data:
        :param min_win_size:
        :param max_win_size:
        :return:
        """
if __name__ == '__main__':
    from IOFactory import IOFactory

    raster = IOFactory.rasterFromAscFile('/home/photo-lab-3/ownCloud/Data/sinkholei11.asc')
    classified, precentMap = ClassificationFactory.ClassifyPoints(raster, 5)
    plt.imshow(classified.ridge)
    plt.show()
