from unittest import TestCase

import matplotlib.pyplot as plt

from IOFactory import IOFactory
from NeighborsFactory import NeighborsFactory
from PanoramaFactory import PanoramaFactory
from SaliencyFactory import SaliencyFactory
from TensorFactory import TensorFactory


class TestSaliencyFactory(TestCase):
    def test_pointwise_tensor_saliency(self):
        pcl = IOFactory.ReadPts('../../IOmodules/test_pts2.pts')
        neighbors = NeighborsFactory.CalculateAllPointsNeighbors(pcl, searchRadius=0.1)
        tensorProp = TensorFactory.computeTensorsProperty_givenNeighborhood(pcl, neighbors)
        saliency = SaliencyFactory.pointwise_tensor_saliency(tensorProp)
        saliencyPanorama = PanoramaFactory.CreatePanorama_byProperty(saliency, elevationSpacing=0.111,
                                                                     azimuthSpacing=0.115, voidData=180)
        plt.imshow(saliencyPanorama.PanoramaImage)
        plt.show()
        self.fail()
