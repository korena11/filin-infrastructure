from unittest import TestCase

from IOmodules.IOFactory import IOFactory
from Properties.Saliency.SaliencyFactory import SaliencyFactory


class TestSaliencyFactory(TestCase):
    # def test_pointwise_tensor_saliency(self):
    #     pcl = IOFactory.ReadPts('../../test_data/test_pts2.pts', merge=False)
    #     neighbors = NeighborsFactory.CalculateAllPointsNeighbors(pcl[0], search_radius=0.1)
    #     tensorProp = TensorFactory.computeTensorsProperty_givenNeighborhood(pcl[0], neighbors)
    #     saliency = SaliencyFactory.pointwise_pca_saliency(tensorProp, verbose=True)
    #     saliencyPanorama = PanoramaFactory.CreatePanorama_byProperty(saliency, elevationSpacing=0.12,
    #                                                                  azimuthSpacing=0.12, voidData=2)
    #     plt.imshow(saliencyPanorama.PanoramaImage)
    #     plt.show()
    #     # self.fail()
    #
    # def test_panorama_frequency(self):
    #     pcl = IOFactory.ReadPts('../../test_data/test_pts2.pts', merge=False)
    #     panorama = PanoramaFactory.CreatePanorama_byPoints(pcl[0], elevationSpacing=0.12,
    #                                                        azimuthSpacing=0.12, voidData=2, intensity=False)
    #     sigma = 2.5
    #     list_sigmas = [sigma, 1.6 * sigma, 1.6 * 2 * sigma, 1.6 * 3 * sigma]
    #     saliency = SaliencyFactory.panorama_frequency(panorama, list_sigmas)
    #     plt.imshow(saliency)
    #     plt.show()
    #
    # def test_panorama_contrast(self):
    #     pcl = IOFactory.ReadPts('../../test_data/test_pts2.pts', merge=False)
    #     panorama = PanoramaFactory.CreatePanorama_byPoints(pcl[0], elevationSpacing=0.12,
    #                                                        azimuthSpacing=0.12, voidData=2, intensity=False)
    #     saliency = SaliencyFactory.panorama_contrast(panorama, region_size=3)
    #     plt.imshow(saliency)
    #     plt.show()
    #
    # def test_panorama_context(self):
    #     pcl = IOFactory.ReadPts('../../test_data/test_pts2.pts', merge=False)
    #     panorama = PanoramaFactory.CreatePanorama_byPoints(pcl[0], elevationSpacing=0.12,
    #                                                        azimuthSpacing=0.12, voidData=2, intensity=False)
    #     saliency = SaliencyFactory.panorama_context(panorama, scale_r=2)
    #     plt.imshow(saliency)
    #     plt.show()

    def test_FPFH_open3d(self):
        from PointSetOpen3D import PointSetOpen3D
        import open3d as o3d

        pcl = IOFactory.ReadPts('../../test_data/test_pts2.pts', merge=True)
        p3d = PointSetOpen3D(pcl)
        knn = o3d.geometry.KDTreeSearchParamKNN(100)
        rad = o3d.geometry.KDTreeSearchParamRadius(radius=0.1)
        hyb = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)

        fpfh = SaliencyFactory.FPFH_open3d(p3d, knn)
        print(fpfh.data)


if __name__ == '__main__':
    from DataClasses.PointSetOpen3D import PointSetOpen3D
    import open3d as o3d

    pcl = IOFactory.ReadPts('../../test_data/test_pts2.pts', merge=True)
    p3d = PointSetOpen3D(pcl)
    knn = o3d.geometry.KDTreeSearchParamKNN(100)
    rad = o3d.geometry.KDTreeSearchParamRadius(radius=0.1)
    hyb = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)

    fpfh = SaliencyFactory.FPFH_open3d(p3d, knn)
    print(fpfh.data)
