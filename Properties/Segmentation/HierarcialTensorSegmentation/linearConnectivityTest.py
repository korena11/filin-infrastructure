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
    surfaceType = 'linear'

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

        ballTree, _labels, nodesOfInterest, tensors = extractSurfaceElements(pntSet,
                                                                            leafSize=10, smallestObjectSize=0.1)

        graph = TensorConnectivityGraph(tensors, numNeighbors=10, varianceThreshold=0.05 ** 2, linearityThreshold=5,
                                        normalSimilarityThreshold=0.01, distanceThreshold=0.08, mode='binary')

        numComponnets, labels, indexesByLabels = graph.connected_componnents()

        if numComponnets > 1:
            print(i, fileList[i])

            linearity = array(list(map(lambda t: t.eigenvalues[-1] / t.eigenvalues[1], tensors)))

            segProp = SegmentationProperty(pntSet, _labels)
            VisualizationO3D.visualize_pointset(segProp)

            graph.spyGraph()

            plt.figure()
            plt.bar(range(len(tensors)), linearity)

            plt.figure()
            plt.imshow(graph._TensorConnectivityGraph__disMatrix.todense(), cmap='gray')
            plt.colorbar()

            plt.figure()
            plt.imshow(graph._TensorConnectivityGraph__donMatrix.todense(), cmap='gray')
            plt.colorbar()
            break

    plt.show()
