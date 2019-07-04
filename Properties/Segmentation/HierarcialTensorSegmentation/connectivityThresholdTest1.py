"""
Code for debugging creation of segmentation property based on Zachi's PhD
DO NOT USE FOR OTHER PURPOSES
"""
from glob import glob
from numpy import zeros, percentile, array, histogram, hstack, round
from scipy.stats import chi2, describe
from scipy.sparse import find
from matplotlib import pyplot as plt

from IOFactory import IOFactory
from SegmentationFactory import SegmentationFactory, SegmentationProperty
from TensorBallTreeSegmentation import tensorConnectedComponents, extractSurfaceElements
from TensorConnectivityGraph import TensorConnectivityGraph
from VisualizationO3D import VisualizationO3D

if __name__ == '__main__':
    path = 'C:/Users/zachis/Dropbox/Research/Code/Segmentation/data/ThresholdLearning/'
    surfaceType = 'smooth'

    fileList = glob(path + surfaceType + "/*/*.pts")
    numFiles = len(fileList)

    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    numPercentiles = len(percentiles)

    disList = []
    percentDisMatrix = zeros((numFiles, numPercentiles))
    descDisList = []
    chi2DisFitRes = zeros((numFiles, 3))
    disTotal = None

    donList = []
    percentDonMatrix = zeros((numFiles, numPercentiles))
    descDonList = []
    chi2DonFitRes = zeros((numFiles, 3))
    donTotal = None

    percentDonMatrix = zeros((numFiles, numPercentiles))

    for i in range(numFiles):
        pntSet = IOFactory.ReadPts(fileList[i])

        ballTree, labels, nodesOfInterest, tensors = extractSurfaceElements(pntSet,
                                                                            leafSize=10, smallestObjectSize=0.1)

        graph = TensorConnectivityGraph(tensors, numNeighbors=10, varianceThreshold=0.01 ** 2,
                                        distanceThreshold=0.1, normalSimilarityThreshold=1e-3)
        rDis, cDis, dis = find(graph._TensorConnectivityGraph__disMatrix)
        rDon, cDon, don = find(graph._TensorConnectivityGraph__donMatrix)

        disTotal = dis if disTotal is None else hstack((disTotal, dis))
        donTotal = don if donTotal is None else hstack((donTotal, don))

        disCounts, disBins = histogram(dis, bins='auto')
        disBins = array(list(map(lambda j: 0.5 * (disBins[j] + disBins[j + 1]), range(disBins.shape[0] - 1))))
        disFreqs = disCounts / disCounts.sum()
        # plt.figure(1)
        # plt.plot(disBins, disFreqs, label=i)

        percentDisMatrix[i] = percentile(dis, percentiles)
        descDisList.append(describe(dis))
        chi2DisFitRes[i] = chi2.fit(dis)

        donCounts, donBins = histogram(don, bins='auto')
        donBins = array(list(map(lambda j: 0.5 * (donBins[j] + donBins[j + 1]), range(donBins.shape[0] - 1))))
        donFreqs = donCounts / donCounts.sum()
        # plt.figure(2)
        # plt.plot(donBins, donFreqs, label=i)

        percentDonMatrix[i] = percentile(don, percentiles)
        descDonList.append(describe(don))
        chi2DonFitRes[i] = chi2.fit(don)

        # print(i, round(descDisList[-1][1], 3), descDonList[-1][1], fileList[i])

    # plt.figure(1)
    # plt.yscale('log')
    # plt.xlabel('Normal Distance [m]')
    # plt.ylabel('Count [-]')
    # plt.legend(loc='best')
    #
    # plt.figure(2)
    # plt.yscale('log')
    # plt.xlabel('Norm of Difference of Normals [-]')
    # plt.ylabel('Count [-]')

    plt.figure()
    plt.hist(disTotal, bins='auto')
    plt.yscale('log')
    plt.xlabel('Normal Distance [m]')
    plt.ylabel('Count [-]')

    plt.figure()
    plt.hist(donTotal, bins='auto')
    plt.yscale('log')
    plt.xlabel('Norm of Difference of Normals [-]')
    plt.ylabel('Count [-]')

    p = [0.7, 0.8, 0.90, 0.95, 0.99]

    disMean = disTotal.mean()
    disStd = disTotal.std()
    disChi2 = chi2.ppf(p, df=1, loc=disMean, scale=disStd)
    print(round(disMean, 3), round(disStd, 3), round(disChi2, 3), '|', round(percentile(disTotal, array(p) * 100), 3))

    donMean = donTotal.mean()
    donStd = donTotal.std()
    donChi2 = chi2.ppf(p, df=1, loc=donMean, scale=donStd)
    print(round(donMean, 3), round(donStd, 3), round(donChi2, 3), '|', round(percentile(donTotal, array(p) * 100), 3))

    from numpy import arccos, pi
    donChi2_deg = arccos((2 - donChi2) / 2) * 180 / pi
    print(donChi2_deg)
    print(arccos((2 - percentile(donTotal, array(p) * 100)) / 2) * 180 / pi)

    plt.show()

    from pickle import dump
    dump(disTotal, open(surfaceType + '_dis.p', 'wb'))
    dump(donTotal, open(surfaceType + '_don.p', 'wb'))
