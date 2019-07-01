from numpy import array, nonzero, unique, hstack, log, exp, isnan, inf, isinf
from numpy.random import choice
from scipy.sparse import coo_matrix

from tqdm import trange


class EnergyBasedSegmentRefiner(object):

    def __init__(self, originalSegmentation, significantSegmentSize=10, searchRadis=None):
        """

        :param originalSegmentation:
        :param significantSegmentSize:
        """

        self.__segmentation = originalSegmentation
        self.__labels = originalSegmentation.GetAllSegments
        tensors = originalSegmentation.segmentAttributes

        # getting the number of tensors composing each segment
        segmentSizes = array(list(map(lambda t: t.tensors_number, tensors)))

        smallSegments = nonzero(segmentSizes <= significantSegmentSize)[0]  # extracting small segments

        self.__pointsOfSmallSegments = hstack(list(map(originalSegmentation.GetSegmentIndices, smallSegments)))
        self.__numPoints = self.__pointsOfSmallSegments.shape[0]

        if searchRadis is None:
            searchRadius = max(list(map(lambda t: t.eigenvalues[-1], tensors[smallSegments]))) ** 0.5 * 100

        self.__pointNeighbors = originalSegmentation.Points.queryRadius(
            originalSegmentation.Points.ToNumpy()[self.__pointsOfSmallSegments], searchRadius)

        self.__dataCosts = coo_matrix((self.__numPoints,
                                       originalSegmentation.NumberOfSegments + 1), dtype='f').tolil()

        self.__segmentNeighbors = list(map(unique, list(map(self.__labels.__getitem__, self.__pointNeighbors))))
        list(map(lambda neighbors: hstack([neighbors, -1]), self.__segmentNeighbors))

        list(map(lambda i: [self.__computeDataCost(i, j) for j in self.__segmentNeighbors[i]],
                 trange(self.__numPoints)))

        self.__energy = self.computeEnergy()

    def __computeDataCost(self, pntIndex, segIndex):
        """

        :param pntIndex:
        :param segIndex:
        :return:
        """
        if self.__dataCosts[pntIndex, segIndex] > 0:
            return

        if segIndex == -1:
            self.__dataCosts[pntIndex, segIndex] = 100  # TODO: define value to junk segment
        else:
            tensor = self.__segmentation.segmentAttributes[segIndex]
            pnt = self.__segmentation.Points.ToNumpy()[self.__pointsOfSmallSegments[pntIndex]]
            self.__dataCosts[pntIndex, segIndex] = tensor.distanceFromPoint(pnt)[0] / tensor.eigenvalues[0] ** 0.5

    def __computeEnergyOfPoint(self, index, label):
        """

        :param index:
        :param label:
        :return:
        """
        # if label != -1:
        if self.__dataCosts[index, label] < 1e-6:
            self.__computeDataCost(index, label)

        #     distAlpha = 10
        #     dist = self.__dataCosts[index, label]
        #     dist /= self.__segmentation.segmentAttributes[label].eigenvalues[0] ** 0.5
        #     if dist > 5:
        #         dist = 5
        #     dataCost = 1 / distAlpha * log((1 + exp(distAlpha * dist)) / (1 + exp(distAlpha * (dist - 1))))
        #     if isnan(dataCost) or isinf(dataCost):
        #         dataCost = 1
        # else:
        #     dataCost = 1

        dataCost = self.__dataCosts[index, label] ** 2

        # smoothCost = nonzero(self.__segmentNeighbors[index] == label)[0].shape[0] / \
        #              self.__pointNeighbors[index].shape[0]
        smoothCost = 0

        return dataCost + smoothCost

    def computeEnergy(self, labels=None):
        if labels is None:
            labels = self.__labels[self.__pointsOfSmallSegments]

        # energies = array(list(map(self.__computeEnergyOfPoint, trange(self.__numPoints), labels)))
        # temp = nonzero(energies == inf)[0]
        # if len(temp) > 1:
        #     a = 1
        return array(list(map(self.__computeEnergyOfPoint, range(self.__numPoints), labels))).sum()

    def optimizeEnergy(self):
        newLabels = array(list(map(choice, self.__segmentNeighbors)))
        newEnergy = self.computeEnergy(newLabels)

        if newEnergy < self.__energy:
            self.__labels[self.__pointsOfSmallSegments] = newLabels
            self.__energy = newEnergy

            self.__segmentNeighbors = list(map(unique, list(map(self.__labels.__getitem__, self.__pointNeighbors))))
            list(map(lambda neighbors: hstack([neighbors, -1]), self.__segmentNeighbors))
            return True
        else:
            return False

    @property
    def energy(self):
        """

        :return:
        """
        return self.__energy


if __name__ == '__main__':
    pass