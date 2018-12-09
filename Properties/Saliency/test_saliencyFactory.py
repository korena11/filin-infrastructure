from unittest import TestCase

from IOFactory import IOFactory
from NeighborsFactory import NeighborsFactory
from SaliencyFactory import SaliencyFactory
from TensorFactory import TensorFactory


class TestSaliencyFactory(TestCase):
    def test_pointwise_tensor_saliency(self):
        pcl = IOFactory.ReadPts('../../IOmodules/test_pts2.pts', merge=False)
        neighbors = NeighborsFactory.CalculateAllPointsNeighbors(pcl[0], searchRadius=0.005)
        tensorProp = TensorFactory.computeTensorsProperty_givenNeighborhood(pcl[0], neighbors)
        saliency = SaliencyFactory.pointwise_tensor_saliency(tensorProp)
        # saliencyPanorama = PanoramaFactory.CreatePanorama_byProperty(saliency, elevationSpacing=0.111,
        #                                                              azimuthSpacing=0.115, voidData=180)
        # plt.imshow(saliencyPanorama.PanoramaImage)
        # plt.show()
        # self.fail()
