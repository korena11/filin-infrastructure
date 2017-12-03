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
from PointSet import PointSet
from EigenFactory import EigenFactory

# Classification codes
RIDGE  =    1
PIT    =    2
VALLEY =    3
FLAT   =    4
PEAK   =    5
SADDLE =    6

class ClassificationFactory:

    @staticmethod
    def ClassifyPoints(data, winsize, **kwargs):
        """
        Classifying points according to their eigenvalues
        :param data: Raster or PointSet data
        :param winsize: the window size according to which teh classification is made
        :param resolution: data resolution (cell size or scanning resolution)
        :param significance level for hypothesis testing. Default: 5%
        :return: a classificaiton map
        """
        #TODO Adjust function for individual points. Now works for raster only

        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            alpha = 0.05

        oneTail = norm.ppf(1-alpha)
        twoTail = norm.ppf(1-alpha/2)


        if isinstance(data, RasterData):
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

            # Hypothesis test for peak:
            # lambda_max < eighThreshold (and therefore lambda_min also)
            # Reject when z_max<-z_1-alpha; Rejected are the peaks
            peak_map = np.zeros(data.shape)
            peak_map[z_max < -oneTail] = PEAK

            # Hypothesis test for ridge:
            #    1.    lambda_min = eighThreshold and lambda_max > eigThreshold
            # or 2.    lambda_max = eighThreshold and lambda_min > eigThreshold

            # 1. reject when  |lambda_min| > z_1-alpha/2 (choose non-rejected) AND when
            #                 lambda_max < -Z_1-alpha (choose reject)
            ridge_map = np.zeros(data.shape)
            ridge_map[(np.abs(z_min) < twoTail) * (z_max < -oneTail)] = RIDGE

            # 2. reject when |lambda_max| > z_1-alpha/2 (choose non-rejected) AND when
            #                lambda_min < -Z_1-alpha (choose reject)
            ridge_map[(np.abs(z_max) < twoTail) * (z_min < -oneTail)] = RIDGE

            # Hypothesis test for valley:
            #    1.    lambda_min = eighThreshold and lambda_max < eigThreshold
            # or 2.    lambda_max = eighThreshold and lambda_min < eigThreshold


            # 1. reject when |lambda_min| > z_1-alpha/2 (choose non-rejected) AND when
            #                lambda_max > Z_1-alpha (choose reject)
            valley_map = np.zeros(data.shape)
            valley_map[(np.abs(z_min) < twoTail) * (z_max > oneTail)] = VALLEY

            # 2. reject when |lambda_max| > z_1-alpha/2 (choose non-rejected) AND when
            #                lambda_min > Z_1-alpha (choose reject)
            valley_map[(np.abs(z_max) < twoTail) * (z_min > oneTail)] = VALLEY

            # Hypothesis test for flat:
            #    lambda_min = eighThreshold and lambda_max = eigThreshold
            flat_map = np.zeros(data.shape)
            flat_map[(np.abs(z_max)< twoTail) * (np.abs(z_min)< twoTail)] = FLAT

            # Hypothesis test for saddle:
            #    lambda_min < eighThreshold and lambda_max > eigThreshold
            saddle_map = np.zeros(data.shape)
            saddle_map[(z_min > oneTail) * (z_max <-oneTail)] = SADDLE

            print 'hello'


if __name__ == '__main__':
    from IOFactory import IOFactory

    raster = IOFactory.rasterFromAscFile('/home/photo-lab-3/ownCloud/Data/sinkholes11.asc')
    ClassificationFactory.ClassifyPoints(raster, 5)