# import mayavi.mlab as mlab
import numpy as np
from numpy import dtype, genfromtxt, nonzero, mean, sum

from Normals.NormalsProperty import NormalsProperty


class NormalsFactory:
    @staticmethod
    def normals_from_file(points, normalsFileName):
        """

        :param points: PointSet to which the normals belong to
        :param normalsFileName: name of file containing normal for each points

        :type points: PointSet.PointSet
        :type normalsFileName: str

        :return: NormalsProperty of the points

        :rtype: NormalsProperty

        """
        
        # Read normals from file                 
        parametersTypes = dtype({'names':['dx', 'dy', 'dz']
                               , 'formats':['float', 'float', 'float']})
        
            
        imported_array = genfromtxt(normalsFileName, dtype=parametersTypes, filling_values=(0, 0, 0))
    
        dxdydz = imported_array[['x', 'y', 'z']].view(float).reshape(len(imported_array), -1)
                            
        normals = NormalsProperty(points, dxdydz)
        
        return normals

    @staticmethod
    def normalsComputation_in_raster(x, y, z):
        r"""
        Computes the normal vectors according where each ordinate is ordered at its place as an ndarray (mxn)
        Usually used for computing normals in panorama images or other rasters.

        According to :cite:`Zeibak2008`, the normals are computed as

        .. math::
            \vec{v}_1 = \begin{bmatrix} dX_1\\ dY_1 \\dZ_1 \end{bmatrix} ; \qquad
            \vec{v}_2 = \begin{bmatrix} dX_2\\ dY_2 \\dZ_2 \end{bmatrix}

        with :math:`dX_i` etc. the differentiation in each direction. Then,

        .. math::

            \vec{N} = \frac{\vec{v}_1\times \vec{v}_2}{||\vec{v}_1\times \vec{v}_2||}

        :param x: x ordinates as organized in the raster
        :param y: y ordinates as organized in the raster
        :param z: z ordinates as organized in the raster

        :return: normals matrices for each direction
        :rtype: np.ndarray [:math:`n\times m \times 3`]
        """
        import MyTools as mt
        # Local derivatives (according to Zeibak p. 56)
        fxx, fxy = mt.computeImageDerivatives(x, order = 2)
        fyx, fyy = mt.computeImageDerivatives(y, order = 1)
        fzx, fzy = mt.computeImageDerivatives(z, order = 1)

        v1 = np.zeros((x.shape[0], x.shape[1], 3))
        v2 = np.zeros((x.shape[0], x.shape[1], 3))

        v1[:, :, 0] = fxx
        v1[:, :, 1] = fyx
        v1[:, :, 2] = fzx

        v2[:, :, 0] = fxy
        v2[:, :, 1] = fyy
        v2[:, :, 2] = fzy

        cross_vec = np.cross(v1, v2, axis = 2)
        n = cross_vec / np.linalg.norm(cross_vec, axis = 2)[:, :, None]

        return n

    @staticmethod
    def __CalcAverageNormal(x, y, z, normalsPoints, normals, eps=0.00001):
        
        indices = nonzero(sum((normalsPoints - [x, y, z]) ** 2, axis=-1) < eps ** 2)[0]
        return mean(normals[indices], axis=0)


#     @staticmethod
#     def VtkNormals(points, triangulation=None):
#         """
#         Calculate normals for each points as average of normals of trianges to which the points belongs to.
#         If no triangulation is given, use TriangulationFactory.Delaunay2D
#
#         :Args:
#
#             - points: PointSet/PointSubSet object
#             - triangulation: triangulationProperty
#
#
#
#         :Returns:
#             - NormalsProperty
#         """
#         polyData = points.ToPolyData
#
#         if triangulation == None:
#             triangulation = TriangulationFactory.Delaunay2D(points)
#
#         polyData.polys = triangulation.TrianglesIndices()
#
#         compute_normals = mlab.pipeline.poly_data_normals(polyData)
#
# #        normals = compute_normals.outputs[0].point_data.normals.to_array()
#
#         mlab.close()
#
#         normals = asarray(map(partial(NormalsFactory.__CalcAverageNormal,
#                                            normalsPoints = compute_normals.outputs[0].points.to_array(),
#                                            normals = compute_normals.outputs[0].point_data.normals.to_array()), points.X, points.Y, points.Z))
#         normals = compute_normals.outputs[0].point_data.normals.to_array()[0 : points.Size]
#
#         return NormalsProperty(points, normals)
#
    
if __name__ == "__main__":
    
    from IOFactory import IOFactory    
    from Visualization import Visualization
    
    pointSetList = []
    
#    IOFactory.ReadXYZ('..\\Sample Data\\cubeSurface.xyz', pointSetList)
#    normalsFileName = '..\\Sample Data\\cubeSurfaceNormals.xyz'
#    normals = NormalsFactory.ReadNormalsFromFile(pointSetList[0], normalsFileName)
    IOFactory.ReadXYZ(r'D:\\Documents\\Pointsets\\cylinder_1.3_Points.txt', pointSetList)
#    triangulation = TriangulationFactory.Delaunay2D(pointSetList[0])
    normals = NormalsFactory.VtkNormals(pointSetList[0])  # , triangulation)
    
    Visualization.RenderPointSet(normals, 'color', color=(0, 0, 0), pointSize=3)
    Visualization.Show()
    
#    points3d(pointSetList[0].X(), pointSetList[0].Y(), pointSetList[0].Z(), scale_factor=.25)
#    quiver3d(pointSetList[0].X(), pointSetList[0].Y(), pointSetList[0].Z(), normals.dX(), normals.dY(), normals.dZ())    
#    show()
        
        
        
         
        
        
