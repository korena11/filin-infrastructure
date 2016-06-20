'''
Created on Nov 26, 2014

@author: Vera
'''

import numpy as np
from NormalsProperty import NormalsProperty
from IOFactory import IOFactory
from Visualization import Visualization
from scipy.spatial import cKDTree
import functools
from NormalsFactory import NormalsFactory
from PointSet import PointSet


def Neighbors(pnt, pntSet, rad=None):
    '''
    Find nearest neighbors of the point in search radius
    :Args:
        - points: pointset
        - pnt: interest point
        - rad: search radius
    :Returns:
        - points: neighbor points
    '''
    if rad == None:
        dist = np.linalg.norm(pnt, ord=2)
        rad = 0.15 * dist / 100  # scan_res*range
    pSize = pntSet.shape[0]
    tree = cKDTree(pntSet)
#     l = tree.query(pnt, 5, p=2)  # , distance_upper_bound=rad)
    l = tree.query(pnt, pSize, p=2, distance_upper_bound=0.07)
    neighbor = l[1][np.where(l[0] != np.inf)[0]]
    points = pntSet[neighbor, :]
    return points


def IncidenceAngle(points):
    '''
    '''
    covMat = np.cov(points.T)  # covariance matrix of pointset
    norm = NormalCalc(covMat)
    meanP = (np.max(points, 0) + np.min(points, 0)) / 2  # mean point
    
    sideP = np.sum((points - np.repeat(np.expand_dims(meanP, 0), points.shape[0], 0)) * np.repeat(norm, points.shape[0], 0), 1)
    side1 = np.where(sideP >= 0)
    side2 = np.where(sideP < 0)

    c1 = np.cov(points[side1[0], :].T)
    N1 = NormalCalc(c1)
    c2 = np.cov(points[side2[0], :].T)
    N2 = NormalCalc(c2)

    norm1 = np.asarray(map(functools.partial(PointNormal, N=N1, points=points), points[side1[0], :]))
    norm2 = np.asarray(map(functools.partial(PointNormal, N=N2, points=points), points[side2[0], :]))
    
    incidenceAng = lambda points, n:np.rad2deg(np.arccos(np.sum(points * n, 1) / (np.linalg.norm(points, 2, 1) * np.linalg.norm(n, 2, 1))))
    ang1 = incidenceAng(points[side1[0], :], norm1)
    ang2 = incidenceAng(points[side2[0], :], norm2)

#     normals = NormalsProperty(PointSet(np.vstack((points[side1[0], :], points[side2[0], :]))), np.vstack((norm1, norm2)))
    normals = NormalsProperty(PointSet(points[side1[0], :]), norm1)
    fig = Visualization.RenderPointSet(normals, renderFlag='color', pointSize=5)
    fig = Visualization.RenderPointSet(PointSet(points[side1[0], :]), renderFlag='color', _figure=fig, color=(0, 1, 0), pointSize=5)
    fig = Visualization.RenderPointSet(PointSet(points[side2[0], :]), renderFlag='color', _figure=fig, color=(0, 0, 1), pointSize=5)
#     Visualization.RenderPointSet(PointSet(meanP), renderFlag='color', _figure=fig, color=(0, 0, 1), pointSize=5)
    Visualization.Show()
    
    
def NormalCalc(covMat):
    '''
    '''
#     normal calculation
    
    eigVal, eigVec = np.linalg.eig(covMat)  # eigVal and eigVec of the covariance matrix 
    norm_ind = np.where(np.abs(eigVal) == np.min(np.abs(eigVal)))
    norm = (eigVec[:, norm_ind]).T[0][0]
    
    return np.expand_dims(norm, 0)


def PointNormal(pnt, N, points):
    '''
    Calculate normals using PCA
    :Args:
        - points: neighbor pointset 
        - pnt: interest point for normal calc.
    :Returns:
        - norm: normal
    '''
    points = Neighbors(pnt, points)
    points = points - np.repeat(np.expand_dims(pnt, 0), points.shape[0], 0)
    C = np.dot(points.T, points) / points.shape[0]  # covariance matrix of pointset
    norm = NormalCalc(C)
    
    alpha = np.arccos(np.sum(N[0] * norm[0]) / (np.linalg.norm(N[0]) * np.linalg.norm(norm[0])))
    
    if alpha > np.pi / 2:
        norm = -norm
    return norm[0]



if __name__ == '__main__':
    print "Loading data..."         
#     (id,x,y,z)
    fileName = 'D:\\Documents\\Pointsets\\cylinder_1.3_Points.txt'  # plane_1.15.txt'
     
    print "Preparing data..."
    points = []
    IOFactory.ReadXYZ(fileName, points)
    points = points[0].ToNumpy
#     d = []
#     for i in xrange(points.shape[0]):
#         for j in xrange(i + 1, points.shape[0]):
#             d.append(np.linalg.norm(points[i, :] - points[j, :]))
#             
#     print np.asarray(d)
    IncidenceAngle(points)
    

