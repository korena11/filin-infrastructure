import numpy as np
from NeighborsFactory import NeighborsFactory
from PCL_registration import *
import time
from scipy.spatial import cKDTree


from IOFactory import IOFactory


def Curvature_FundamentalForm_GPU(pnt, points, rad):
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


def Curvature_FundamentalForm_NUMPY(pnt, points, rad):
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
    neighbor = NeighborsFactory.GetNeighborsIn3dRange_KDtree(pnt, points, rad, tree)
    neighbors = neighbor.ToNumpy()
    if np.greater(np.shape(neighbors[1::, :])[0] , 5):
        neighbors = np.subtract(neighbors, pnt)[1::, :]
        eigVal, eigVec = PCA(neighbors, rad)
        
        normP = eigVec[:, np.where(np.equal(eigVal, np.min(eigVal)))[0][0]]
        # if a normal of a neighborhood is in an opposite direction rotate it 180 degrees
        if np.less(np.linalg.norm(pnt, 2), np.linalg.norm(pnt + normP, 2)):
            normP = -normP
        n = np.array([0, 0, 1])
        
        # rotate the neighborhood to xy plane 
        rot_mat = Rotation_Matrix(normP, n)
        
        neighbors = np.transpose(np.dot(rot_mat, np.transpose(neighbors)))
        pnt = np.array([0, 0, 0])
        pQuality = GoodPoint(neighbors, rad)
        
    #     fig1 = Visualization.RenderPointSet(PointSet(points), renderFlag='height', pointSize=4)
    #     Visualization.RenderPointSet(PointSet(np.expand_dims(pnt, 0)), renderFlag='color', _figure=fig1, color=(0, 0, 0), pointSize=6)
    #     Visualization.Show()
        if  np.equal(pQuality, 1):
            p = BiQuadratic_Surface(np.vstack((neighbors, pnt)))
            Zxx = np.multiply(2, p[0])
            Zyy = np.multiply(2, p[1])
            Zxy = p[2]
         
            k1 = (np.divide
                  (np.add
                   (np.add
                     (Zxx, Zyy), (np.sqrt
                                  (np.add
                                   (np.power(np.subtract(Zxx, Zyy), 2) , np.multiply(4, np.power(Zxy, 2)))
                                   ))), 2))[0]
            k2 = (np.divide(np.subtract(np.add(Zxx, Zyy), (np.sqrt(np.add(np.power(np.subtract(Zxx, Zyy), 2) , np.multiply(4, np.power(Zxy, 2)))))), 2))[0]
        else:
            k1, k2 = -999, -999
    else:
        k1, k2 = -999, -999
    
    return np.array([k1, k2])



def Curvature_FundamentalForm(pnt, points, rad):
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

path = 'D:\\Documents\\Pointsets\\'
fileName = path + 'Bonim_set1.pts'

pointSet = []
IOFactory.ReadPts(fileName, pointSet)
pointSet = pointSet[0]
pp = pointSet.ToNumpy()
tree = cKDTree(pp)
print pointSet.Size

 
begin = time.time()
# [Curvature_FundamentalForm_NUMPY(pp[2 * i, :], pointSet, 0.6) for i in xrange(10)]

for i in xrange(100):
    Curvature_FundamentalForm_NUMPY(pp[ i, :], pointSet, 0.6)

end = time.time()
print "%.6f" % ((end - begin))

begin = time.time()
# [Curvature_FundamentalForm(pp[2 * i, :], pointSet, 0.6) for i in xrange(10)]
for i in xrange(100):
    Curvature_FundamentalForm(pp[i, :], pointSet, 0.6)
end = time.time()
print "%.6f" % ((end - begin))

begin = time.time()
# [Curvature_FundamentalForm(pp[2 * i, :], pointSet, 0.6) for i in xrange(10)]
for i in xrange(100):
    Curvature_FundamentalForm(pp[i, :], pointSet, 0.6)
end = time.time()
print "%.6f" % ((end - begin))
 
begin = time.time()
# [Curvature_FundamentalForm_NUMPY(pp[2 * i, :], pointSet, 0.6) for i in xrange(10)]

for i in xrange(100):
    Curvature_FundamentalForm_NUMPY(pp[ i, :], pointSet, 0.6)

end = time.time()
print "%.7f" % ((end - begin))


