from unittest import TestCase

from matplotlib import pyplot as plt

from IOFactory import IOFactory
from NeighborsFactory import NeighborsFactory
from PanoramaFactory import PanoramaFactory
from SaliencyFactory import SaliencyFactory
from TensorFactory import TensorFactory


class TestSaliencyFactory(TestCase):
    def test_pointwise_tensor_saliency(self):
        pcl = IOFactory.ReadPts('../../test_data/test_pts2.pts', merge=False)
        neighbors = NeighborsFactory.CalculateAllPointsNeighbors(pcl[0], searchRadius=0.1)
        tensorProp = TensorFactory.computeTensorsProperty_givenNeighborhood(pcl[0], neighbors)
        saliency = SaliencyFactory.pointwise_tensor_saliency(tensorProp, verbose=True)
        saliencyPanorama = PanoramaFactory.CreatePanorama_byProperty(saliency, elevationSpacing=0.12,
                                                                     azimuthSpacing=0.12, voidData=2)
        plt.imshow(saliencyPanorama.PanoramaImage)
        plt.show()
        # self.fail()
