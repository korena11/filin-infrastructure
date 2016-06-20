from PC_registration import *
from numbapro import float64, float32, int32, int64, jit, vectorize, bool_, autojit, cuda
import numpy as np
import time
from PointSet import PointSet
from NeighborsFactory import NeighborsFactory
from IOFactory import IOFactory

def Negetive(vec):
    return -vec



def Curvature_FundamentalForm(pnt, pointset, rad):
    '''
    curvature computation based on fundamental form
    :Args:
        - pnt: array 3x1 point of interest
        - points: pointset
        - rad: radius of the neighborhood
    :Returns:
        - principal curvatures
    '''
    # find point's neighbors in a radius
    points = PointSet(pointset)
    neighbor = NeighborsFactory.GetNeighborsIn3dRange_KDtree(pnt, points, rad, tree)
    neighbors = neighbor.ToNumpy()
    if neighbors[1::, :].shape[0] > 5:
        neighbors = (neighbors - pnt)[1::, :]
        eigVal, eigVec = PCA(neighbors, rad)
        
        normP = eigVec[:, np.where(eigVal == np.min(eigVal))[0][0]]
        # if a normal of a neighborhood is in an opposite direction rotate it 180 degrees
        if np.linalg.norm(pnt, 2) < np.linalg.norm(pnt + normP, 2):
            normP = -normP
        n = np.array([0, 0, 1])
        
        # rotate the neighborhood to xy plane 
        rot_mat = Rotation_Matrix(normP, n)
        
        neighbors = (np.dot(rot_mat, neighbors.T)).T
        pnt = np.array([0, 0, 0])
        pQuality = GoodPoint(neighbors, rad)
        
    #     fig1 = Visualization.RenderPointSet(PointSet(points), renderFlag='height', pointSize=4)
    #     Visualization.RenderPointSet(PointSet(np.expand_dims(pnt, 0)), renderFlag='color', _figure=fig1, color=(0, 0, 0), pointSize=6)
    #     Visualization.Show()
        if  pQuality == 1:
            p = BiQuadratic_Surface(np.vstack((neighbors, pnt)))
            Zxx = 2 * p[0]
            Zyy = 2 * p[1]
            Zxy = p[2]
         
            k1 = (((Zxx + Zyy) + np.sqrt((Zxx - Zyy) ** 2 + 4 * Zxy ** 2)) / 2)[0]
            k2 = (((Zxx + Zyy) - np.sqrt((Zxx - Zyy) ** 2 + 4 * Zxy ** 2)) / 2)[0]  
        else:
            k1, k2 = -999, -999
    else:
        k1, k2 = -999, -999
    
    return np.array([k1, k2])



class GpuFuncs:
    
    def __init__(self):
        pass
    
    
    @staticmethod
    @vectorize(['float64(float64,float64)', 'float64(float32,float32)'], target='gpu')
    def GPU(vec1, vec2):
        return a + b
    
    @staticmethod
    @jit(target='gpu')
    # @cuda.jit(argtypes=[float64[:], float64[:], float64[:], float64[:, :], float64], target='gpu')
    def curvatureLoopGPU(k1, k2, pnt, points, radius):
        i = cuda.grid(1)
        tmpCurvatureVec = Curvature_FundamentalForm(points[pnt[i], :], points, rad)
        k1[i] = tmpCurvatureVec[0]
        k2[i] = tmpCurvatureVec[1]
    
# arr1 = 5.2 * np.ones((0.5e6, 1))
# arr2 = 4 * np.ones((0.5e6, 1))
 
arr1 = np.ones((3e6, 1))
arr2 = np.ones((3e6, 1))
# pnt = xrange(10) 

begin = time.time()
np.sqrt(arr1) + np.sqrt(arr2)
end = time.time()
print "%.8f" % ((end - begin))
  
 
begin = time.time()
GpuFuncs.GPU(arr1, arr2)
end = time.time()
  
  
  
print "%.8f" % ((end - begin))

