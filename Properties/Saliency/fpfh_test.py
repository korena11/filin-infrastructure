import open3d as o3d

from DataClasses.PointSetOpen3D import PointSetOpen3D
from IOmodules.IOFactory import IOFactory

pcl = IOFactory.ReadPts('../../test_data/test_pts.pts', merge=True)
p3d = PointSetOpen3D(pcl)
# knn = o3d.geometry.KDTreeSearchParamKNN(100)
# rad = o3d.geometry.KDTreeSearchParamRadius(radius=0.1)
# hyb = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)

pcd = o3d.io.read_point_cloud('../../test_data/test_pts2.pts')

fpfh_o3d = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=100))

# fpfh = SaliencyFactory.FPFH_open3d(p3d, knn)
# print(fpfh.data)
print('hello')
