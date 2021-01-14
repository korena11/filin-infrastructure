
from DataClasses.KdTreePointSet import KdTreePointSet
from DataClasses.PointSetOpen3D import PointSetOpen3D
from IOmodules.IOFactory import IOFactory
from Properties.Curvature.CurvatureFactory import CurvatureFactory
from Properties.Neighborhood.NeighborsFactory import NeighborsFactory
from Properties.Normals.NormalsFactory import NormalsFactory
from VisualizationClasses.VisualizationO3D import VisualizationO3D

if __name__ == '__main__':
    ptset = IOFactory.ReadPts(r'../../test_data/wave.pts')
    o3d = PointSetOpen3D(ptset)
    v3d = VisualizationO3D()
    kdset = KdTreePointSet(ptset)
    # v3d.visualize_pointset(o3d)
    neighborhood = NeighborsFactory.kdtreePointSet_rnn(kdset, 0.5)
    normals, o3d = NormalsFactory.normals_open3D(o3d, 0.5)

    curvatureProperty = CurvatureFactory.pointSetOpen3D_3parameters(o3d, neighborhood, valid_sectors=4, invalid_value=0)
    curvatureProperty = CurvatureFactory.umbrella_curvature(neighborhood,normals,curvatureProperty=curvatureProperty, invalid_value=0)
    v3d.visualize_property(curvatureProperty)

