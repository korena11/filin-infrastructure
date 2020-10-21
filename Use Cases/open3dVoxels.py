from IOFactory import IOFactory
from PointSetOpen3D import PointSetOpen3D
import open3d as o3d
from VisualizationO3D import VisualizationO3D

if __name__ == '__main__':
    path = '../test_data/'
    filename = 'test_ply'
    pntSet = IOFactory.ReadPly(path + filename + '.ply', returnAdditionalAttributes=False)

    pntSet = PointSetOpen3D(pntSet.ToNumpy())
    # VisualizationO3D.visualize_pointset(pntSet)

    tmp = o3d.voxel_down_sample_and_trace(pntSet.data, 0.1,
                                          pntSet.ToNumpy().min(axis=0).reshape((3, 1)) - 1e-6,
                                          pntSet.ToNumpy().max(axis=0).reshape((3, 1)) + 1e-6, approximate_class=True)
    print("Done!")
