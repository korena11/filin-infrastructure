'''
Created on Dec 2, 2014

@author: Vera
'''

import datetime
import functools
import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hac
import scipy.spatial.distance as distance
from matplotlib.mlab import PCA
from numba import jit, double, i4
from scipy import signal
from scipy import spatial
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

from IOFactory import IOFactory
from NeighborsFactory import NeighborsFactory
from PointSet import PointSet
from PointSubSet import PointSubSet
from SphericalCoordinatesFactory import SphericalCoordinatesFactory
from TriangulationFactory import TriangulationFactory


def Curvature_Euler(pnt, pntNormal, coeff, points_normals):
    '''

    according to: 2008 - Zhang etal. Curvature estimation of 3D point cloud surfaces through the fitting of normal curvatures.
    :Args:
        - pnt: array 3x1 point of interest
        - pntNormal: array 3x1 points of interest normal
        - coeff: coefficient for radius calc
        - points_normals: NormalProperty (points, normals)
    :Returns:
        - H: mean curvature
    '''
    rad = coeff * 0.10 * (np.linalg.norm(pnt, 2)) / 100
    points = points_normals.Points
    normals = points_normals.Normals
     
    points = NeighborsFactory.GetNeighborsIn3dRange_KDtree(pnt, points, rad)
    indices = points.GetIndices
    normals = normals[indices[1::], :]
    
#     fig = Visualization.RenderPointSet(points, renderFlag='color', pointSize=5)
#     fig = Visualization.RenderPointSet(PointSet(np.expand_dims(pnt, 0)), renderFlag='color', _figure=fig, color=(1, 0, 0), pointSize=10)
#     Visualization.Show()
     
#     transformation to local coord. system
    
    if points.Size > 1:
        points = points.ToNumpy()
#         if np.linalg.norm(pnt, 2) < np.linalg.norm(pnt + normP, 2):
#             normP = -normP
#             
#         rot_mat = Rotation_Matrix(normP, np.array([0, 0, 1]))
        points = (points - np.repeat(np.expand_dims(pnt, 0), points.shape[0], 0))[1::, :]
#         points = (np.dot(rot_mat, points.T)).T
#         pntNormal = np.array([0, 0, 1])
#         normals = (np.dot(rot_mat, normals.T)).T
        
        psi = np.arccos(pntNormal[2])
        phi = np.arctan(pntNormal[1] / pntNormal[0])
        X = np.array([-np.sin(phi), np.cos(phi), 0])
        Y = np.array([np.cos(psi) * np.cos(phi), np.cos(psi) * np.sin(phi), -np.sin(psi)])
     
    #     normal curvature calculation
        n_xy = (points[:, 0] * normals[:, 0] + points[:, 1] * normals[:, 1]) / np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        kn = -n_xy / (np.sqrt(n_xy ** 2 + normals[:, 2] ** 2) * np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2))
        x_X = np.arctan2(X[1], X[0])
        x_q = np.arctan2(points[:, 1], points[:, 0])
        theta_i = np.abs(np.repeat(np.expand_dims(x_X, 0), points.shape[0], 0) - x_q)
#         theta_i = np.arccos(np.sum(points * np.repeat(np.expand_dims(X, 0), points.shape[0], 0), 1) / (np.linalg.norm(points, 2, 1) * np.linalg.norm(X, 2)))
    
        if np.sum(np.isnan(n_xy)):
            ind = np.where(np.isnan(n_xy) == True)
            kn = np.delete(kn, ind)
            theta_i = np.delete(theta_i, ind)
        
    #     ==============principal curvatures==============
        ct = np.cos(theta_i)
        st = np.sin(theta_i)
        
    #     Least Squares
        A = np.hstack((np.expand_dims(ct ** 2, 1), np.expand_dims(2 * ct * st, 1), np.expand_dims(st ** 2, 1)))
        x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, np.expand_dims(kn, 1)))
        k, _ = np.linalg.eig(np.array([[x[0, 0], x[1, 0]], [x[1, 0], x[2, 0]]]))  # k = [k1 k2] - principle curvatures
    else:
        k = [0, 0]
    return np.asarray(k)
    

def Curvature_ExternalNeighbor(pnt, points, coeff):
    '''
    absolute curvature computation based on triangles' normals
    :Args:
        - pnt: array 3x1 point of interest
        - pntNormal: array 3x1 points of interest normal
        - coeff: coefficient for radius calc
    :Returns:
        - H: mean curvature
    '''
    rad = coeff * 0.10 * (np.linalg.norm(pnt, 2)) / 100
    points = (NeighborsFactory.GetNeighborsIn3dRange_KDtree(pnt, points, rad)).ToNumpy()
    
    convex_hull = spatial.ConvexHull(points[:, 0:2])
    conv_vertices = convex_hull.vertices
    
#     fig = Visualization.RenderPointSet(PointSet(points), renderFlag='color', pointSize=5)
#     fig = Visualization.RenderPointSet(PointSet(np.expand_dims(pnt, 0)), renderFlag='color', _figure=fig, color=(1, 0, 0), pointSize=10)
#     Visualization.RenderPointSet(PointSet(points[conv_vertices, :]), renderFlag='color', _figure=fig, color=(0, 1, 0), pointSize=7)
#     mlab.plot3d(points[conv_vertices, 0], points[conv_vertices, 1], points[conv_vertices, 2], tube_radius=0.001, colormap='gray')
#     Visualization.Show()
    
    normal = lambda(ed):(np.cross(ed[0:-1, :], ed[1::, :], 1)) / np.expand_dims(np.linalg.norm((np.cross(ed[0:-1, :], ed[1::, :], 1)), 2, 1), 0).T
    norm_angle = lambda(n):np.arccos(np.sum(n[0:-1, :] * n[1::, :], 1) / (np.linalg.norm(n[0:-1, :], 2, 1) * np.linalg.norm(n[1::, :], 2, 1)))
    
    if 0 in conv_vertices:
        ind = np.where(conv_vertices == 0)[0][0]
        conv_vertices = np.hstack((conv_vertices[ind + 1::], conv_vertices[0:ind]))
        edges = points[conv_vertices, :] - points[0, :]
        
        tri_normal = normal(edges)
        angle = norm_angle(tri_normal)
        curv = 0.25 * (np.sum(np.linalg.norm(edges[1:-1, :], 2, 1) * np.abs(angle)))
#         print curv
    else: 
        conv_vertices = np.hstack((conv_vertices, conv_vertices[0]))
        edges = points[conv_vertices[0:-1], :] - points[0, :]
        
        tri_normal = normal(np.vstack((edges, edges[0, :])))
        angle = norm_angle(np.vstack((tri_normal[-1, :], tri_normal)))
        curv = 0.25 * (np.sum(np.linalg.norm(edges, 2, 1) * np.abs(angle)))
#         print curv

    return curv


def SurfVariation(pnt, points, coeff):
    '''
    Compute surface variation
    :Args:
        - pnt: point of interest
        - points: pointset
        - num_neighbors: number of neighbors to use in calculation
    :Returns:
        - SurfVar: surface variation value
    '''
    rad = coeff * 0.10 * (np.linalg.norm(pnt, 2)) / 100
#     rad = 0.20
    points = (NeighborsFactory.GetNeighborsIn3dRange_KDtree(pnt, points, rad)).ToNumpy()
    
    eigVal, _ = PCA(pnt, points)
    eigVal = eigVal[::-1]
    surfVar = eigVal[2] / np.sum(eigVal)
                
    return surfVar  


def PointCharacter_123D(eigVal):
    '''
    determine points character - line/plane/spatial
    '''
    denom = np.sum(eigVal)
    a1 = eigVal[0] / denom
    a2 = eigVal[1] / denom
    a3 = eigVal[2] / denom
     
#     sigma = np.sqrt(eigVal)
#     a1 = (sigma[0] - sigma[1]) / sigma[0]
#     a2 = (sigma[1] - sigma[2]) / sigma[0]
#     a3 = sigma[2] / sigma[0]
 
#     if (np.abs(np.max(eigVal[0:2]) / np.min(eigVal[0:2])) >= 1 and 
#         np.abs(np.max(eigVal[0:2]) / np.min(eigVal[0:2])) < 10 and 
#         np.abs(np.max(eigVal[0:2]) / eigVal[2]) >= 10):
#     if (a1 > a2 and a1 > a3):
    if (a1 / a2 < 10 and a1 / a3 < 10):
        c = np.array([0, 255, 0])  # point
    else:
#         if (np.abs(eigVal[0] / eigVal[1]) >= 10 and 
#             np.abs(eigVal[0] / eigVal[2]) >= 10):
#         if (a2 > a1 and a2 > a3):
        if (a1 > 0.6):
            c = np.array([0, 0, 255])  # line
        else:
            c = np.array([255, 0, 0])  # plane
            
    return c


def Rotation_Matrix(a1, a2):
    '''
    compute rotation matrix from vector a1 to a2
    :Args:
        - a1,a2: vectors
    :Returns:
        - R: rotation matrix
    '''
    v = np.cross(a1, a2)
    c = np.sum(a1 * a2)
    h = (1 - c) / (1 - c ** 2)
    
    R = np.array([[c + h * v[0] ** 2, h * v[0] * v[1] - v[2], h * v[0] * v[2] + v[1]],
                 [h * v[0] * v[1] + v[2], c + h * v[1] ** 2, h * v[1] * v[2] - v[0]],
                 [h * v[0] * v[2] - v[1], h * v[1] * v[2] + v[0], c + h * v[2] ** 2]])

    return R


@jit(double(double[:, :], double))
def PCA(points, rad=None):
    '''
    compute eigenvalues and eigenvectors
    :Args: 
        - points: ndarray nx3 points with origin in pnt
    '''    
    
    if rad != None:
        sigma = rad / 3
        w = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2) / (2 * sigma ** 2))
        if np.sum(np.isnan(w)) > 0 or np.sum(np.isinf(w)) > 0 or np.abs(np.sum(w)) < 1e-10:
            w = np.ones(points[:, 0].shape)
    else: w = np.ones(points[:, 0].shape)
    
    pT = np.vstack((np.expand_dims(w * points[:, 0], 0), np.expand_dims(w * points[:, 1], 0), np.expand_dims(w * points[:, 2], 0)))
    
    C = np.dot(pT, points) / np.sum(w)  # points.shape[0]  # covariance matrix of pointset
    eigVal, eigVec = np.linalg.eig(C)
    
    return eigVal, eigVec
 

def GaussianSmooth(p, points, res, coeff, sigma_arg=1):
    '''
    Gaussian smoothing
    :Args:
        - p: point of interest
        - points: pointset - in spherical coordinates
        - res: angular residual in radians or cm (x cm by 100 m)
        - sigma_arg: argument that multiplies sigma to change the gaussian size
    :Returns:
        - range: new range value
    '''
    rad = sigma_arg * np.arctan(coeff * res * p[2] / 100 ** 2)
    
    
    sigma = rad / 2.5
    points = NeighborsFactory.GetNeighborsIn3dRange_SphericCoord(p, points, rad)
    G = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(np.deg2rad(points.X - p[0]) ** 2 + np.deg2rad(points.Y - p[1]) ** 2) / (2 * sigma ** 2))
    norm_G = G / np.sum(G)
    range = np.sum(points.Z * norm_G)
    return range
    

def Normal(pnt, points, coeff, pntTree):
    '''
    normal vector calculation using PCA
    :Args:
        - pnt: the point of interest
        - points: points
        - coeff: scale coefficient
    :Returns:
        - norm: normal vector in the given point
    '''
    rad = coeff * 0.10 * (np.linalg.norm(pnt, 2)) / 100
    points = (NeighborsFactory.GetNeighborsIn3dRange_KDtree(pnt, points, rad, tree=pntTree)).ToNumpy()
    if points.shape[0] > 5:
        eigVal, eigVec = PCA(pnt, points, rad)
        norm_ind = np.where(eigVal == np.min(eigVal))
        norm = (eigVec[:, norm_ind]).T[0][0]
    else:
        norm = np.array([0, 0, 0])
    
#     if np.linalg.norm(pnt, 2) < np.linalg.norm(pnt + norm, 2):
#         norm = -norm
    
    return norm



def BiQuadratic_Surface(points):
    '''
    BiQuadratic surface adjustment to discrete point cloud
    :Args: 
        - points: 3D points coordinates
    :Returns:
        - p: surface's coefficients
    '''
    # ==== initial guess ====
    x = np.expand_dims(points[:, 0], 1)
    y = np.expand_dims(points[:, 1], 1)
    z = np.expand_dims(points[:, 2], 1)
    
    A = np.hstack((x ** 2, y ** 2, x * y, x, y, np.ones(x.shape)))
    p = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, z))
    
    return p

  
# ======================== Curvature Functions ===============================
 

def GoodPoint(points, rad):
    '''
    determine weather the point is appropriate for curvature calculation
    '''
#     fig = Visualization.RenderPointSet(PointSet(points), renderFlag='color', color=(0, 0, 0), pointSize=3)
#     Visualization.RenderPointSet(PointSet(np.array([[0, 0, 0]])), renderFlag='color', _figure=fig, color=(1, 0, 0), pointSize=4)
#     Visualization.Show()

    sector = 0
    pAngle = np.zeros((1, points.shape[0]))
    ind1 = np.where(np.abs(points[:, 0]) > 1e-6)[0]
    pAngle[0, ind1] = np.arctan2(points[ind1, 1], points[ind1, 0])
    ind2 = np.where(np.abs(points[:, 0]) <= 1e-6)[0]
    if ind2.size != 0:
        ind2_1 = np.where(points[ind2, 1] > 0)[0]
        ind2_2 = np.where(points[ind2, 1] < 0)[0]
        if ind2_1.size != 0:
            pAngle[0, ind2[ind2_1]] = np.pi / 2.0
        if ind2_2.size != 0:
            pAngle[0, ind2[ind2_2]] = 3.0 * np.pi / 2.0
    
    pAngle[np.where(pAngle < 0)] += 2 * np.pi
    for i in np.linspace(0, 7.0 * np.pi / 4.0, 8):
        pInSector = (np.where(pAngle[np.where(pAngle <= i + np.pi / 4.0)] > i))[0].size
        if pInSector >= rad * 85:
            sector += 1
    
#     print sector        
    if sector >= 7:
        return 1
    else:
        return 0


@jit(double(double[:], double, double))
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
#     rad = coeff * 0.10 * (np.linalg.norm(pnt, 2)) / 100
#     fig = Visualization.RenderPointSet(points, renderFlag='height', pointSize=4)
    # find point's neighbors in radius
    
#     Visualization.RenderPointSet(PointSet(points), renderFlag='color', _figure=fig, color=(0, 0, 0), pointSize=6)
#     Visualization.RenderPointSet(PointSet(np.expand_dims(pnt, 0)), renderFlag='color', _figure=fig, color=(1, 0, 0), pointSize=6)
#     Visualization.Show()
    
    neighbor = NeighborsFactory.GetNeighborsIn3dRange_KDtree(pnt, points, rad, tree)
    neighbors = neighbor.ToNumpy()
    if neighbors[1::, :].shape[0] > 5:
        neighbors = (neighbors - np.repeat(np.expand_dims(pnt, 0), neighbors.shape[0], 0))[1::, :]
        eigVal, eigVec = PCA(neighbors, rad)
        
        normP = eigVec[:, np.where(eigVal == np.min(eigVal))[0][0]]
        if np.linalg.norm(pnt, 2) < np.linalg.norm(pnt + normP, 2):
            normP = -normP
    #         n = np.array([0, 0, -1])
    #     else:
    #         n = np.array([0, 0, 1])
        n = np.array([0, 0, 1])
         
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


def DataSnooping(data, numOfSTD):
    '''
    values out of the range [mean-2.5std, mean+2.5std] 
    :Args:
        - data: data nx1 array
        - numOfSTD: number of std that items bigger\less than (mean +\- numOfSTD*std) are out of range of data
    :Returns:
        - ind: indices of points that "survived"
    '''
    stdData = np.std(data)
    meanData = np.mean(data)
    ind1 = np.where(data > meanData - numOfSTD * stdData)
    data = data[ind1]
    ind2 = np.where(data < meanData + numOfSTD * stdData)
    ind = (ind1[0])[ind2]
    return ind


def Similarity_Curvature(k1, k2):
    '''
    calculates similarity curvature (E,H)
    :Args:
        - k1,k2: principal curvatures (k1>k2)
    :Returns:
        - similarCurv: values of similarity curvature (E,H)
        - rgb: RGB color for every point
    '''
    
    k3 = np.min((np.abs(k1), np.abs(k2)), 0) / np.max((np.abs(k1), np.abs(k2)), 0)
    similarCurv = np.zeros((k3.shape[0], 2))
    rgb = np.zeros((k3.shape[0], 3), dtype=np.float32)
    
    sign_k1 = np.sign(k1)
    sign_k2 = np.sign(k2)
    signK = sign_k1 + sign_k2
    
    # (+,0) 
    positive = np.where(signK == 2)
    similarCurv[positive[0], 0] = k3[positive]
    rgb[positive[0], 0] = k3[positive]
    # (-,0)
    negative = np.where(signK == -2)
    similarCurv[negative[0], 0] = -k3[negative]
    rgb[negative[0], 1] = k3[negative]
    
    dif = (np.where(signK == 0))[0]
    valueK = np.abs(k1[dif]) >= np.abs(k2[dif])
    k2_k1 = np.where(valueK == 1)
    k1_k2 = np.where(valueK == 0)
    # (0,+)
    similarCurv[dif[k2_k1[0]], 1] = (k3[dif[k2_k1[0]]].T)[0]
    rgb[dif[k2_k1[0]], 0:2] = np.hstack((k3[dif[k2_k1[0]]], k3[dif[k2_k1[0]]]))
    # (0,-)
    similarCurv[dif[k1_k2[0]], 1] = -(k3[dif[k1_k2[0]]].T)[0]
    rgb[dif[k1_k2[0]], 2] = (k3[dif[k1_k2[0]]].T)[0]
    
    return rgb, similarCurv


@jit(double(i4, double[:, :], i4[:], double[:, :], double))
def ZeroCrossing_Curvature(iP, points, iNegative, meanCurv, tree):
    '''
    find zero-crossing of the curvature and its magnitude
    :Args:
        - iP: index of the point
        - points: PointSet object
        - iNegative: indices of the points with negative mean curvature
        - meanCurv: mean curvatures of the pointset
    :Returns:
        - magnitude of zero-crossing
    '''
    pnt = points[iP, :]
    neighbors = NeighborsFactory.GetNeighborsIn3dRange_KDtree(pnt, PointSet(points), 0.05, tree)

#     neighbors = (NeighborsFactory.GetNeighborsIn3dRange_BallTree(pnt, points, 0.05))
    difCurv = 0
    if neighbors.Size > 3:
        pnts = neighbors.GetIndices
        neighbors = neighbors.ToNumpy()
        neighbors = neighbors - np.repeat(np.expand_dims(pnt, 0), neighbors.shape[0], 0)
        eigVal, eigVec = PCA(neighbors[1::, :], 0.05)
            
        normP = eigVec[:, np.where(eigVal == np.min(eigVal))[0][0]]
        n = np.array([0, 0, 1])
         
        rot_mat = Rotation_Matrix(normP, n)
        neighbors = (np.dot(rot_mat, neighbors.T)).T
    
        triangulation = (Delaunay(neighbors[:, 0:2])).simplices
        tri = np.where(triangulation == 0)[0]
         
        for i in tri:
            for j in xrange(3):  
                if np.sum(iNegative == pnts[(triangulation[i])[j]]) > 0:
                    difCurv = meanCurv[iP] - meanCurv[pnts[(triangulation[i])[j]]]
                    return difCurv
    return difCurv



def CurveClusters(i, points, maxClust=None, clusters=None, fig=None):
    '''
    given pointset construct curve segments
    :Args:
        - points: PointSet
    :Returns:
    '''
    if i == 0 and clusters == None:
        points = points.ToNumpy()
    else:
        points = points.ToNumpy()[(clusters == i), :]
        
    if maxClust == None:
        maxClust = np.sqrt(points.shape[0] / 6)
    
    if points.shape[0] > 1:
        method = 'single'
        metric = 'cosine'  # 'euclidean'  # 
        
        z = hac.linkage(points, method=method, metric=metric)
        
        part = hac.fcluster(z, maxClust, 'maxclust')
        
        print 'number of clusters : ', np.max(part) 
        
        if fig != None:
            fig = Visualization.RenderPointSet(PointSet(points), renderFlag='parametericColor', _figure=fig,
                                            color=(part, np.zeros(part.size), np.zeros(part.size)), pointSize=3, colorMap='Paired', colorBar=1)
        else:
            Visualization.RenderPointSet(PointSet(points), renderFlag='parametericColor',
                                color=(part, np.zeros(part.size), np.zeros(part.size)), pointSize=3, colorMap='Paired', colorBar=1)
    else:
        part = np.array([1])
    return part  # , points


def TriNeighbors(i, tri, ind):
    '''
    finds neighbors from first and second triangles' rings
    :Args:
        - i: point index
        - tri: triangulation - points indices in every triangle
        - ind: indices of points in a segment
    :Returns:
        - firstRingNeighbors: first ring neighbors (ndarray)
        - secondRingNeighbors: second ring neighbors (ndarray)
    '''
    triangles = (np.where(tri == i))[0]
    trNeighbors = tri[triangles, :]
    firstRingNeighbors = np.asarray(trNeighbors[trNeighbors != i])
    firstRingNeighbors = np.unique(firstRingNeighbors, return_index=True)[0]
    
    secondRingNeighbors = []
    for p in firstRingNeighbors:
        triangles = (np.where(tri == p))[0]
        trNeighbors = tri[triangles, :]
        neighbors = np.delete(trNeighbors, (np.where(trNeighbors == i))[0], axis=0)
        secondRingNeighbors.append(neighbors.tolist())    
    secondRingNeighbors = np.unique(np.asarray(list(itertools.chain.from_iterable(secondRingNeighbors))), return_index=True)[0]
    
    # leave neighbors that are part of the cluster
    i = 0
    while i < len(firstRingNeighbors):
        if firstRingNeighbors[i] not in ind:
            firstRingNeighbors = np.delete(firstRingNeighbors, i, 0)
        else:
            i += 1
    i = 0
    while i < len(secondRingNeighbors):
        if secondRingNeighbors[i] not in ind or secondRingNeighbors[i] in firstRingNeighbors:
            secondRingNeighbors = np.delete(secondRingNeighbors, i, 0)
        else:
            i += 1
    return firstRingNeighbors, secondRingNeighbors
    

def CurveSegments(points, indices, tri, clust_ID):
    '''
    cluster refinement according to tringulation neighbors
    :Args:
        - points: (nX3 ndarray) all points coordinates
        - indices: (nX1 ndarray) indices of points in a segment
        - tri: (nX3 ndarray) triangulation indices
        - clust_ID: (int) index of the last updated cluster 
    :Returns:
        - clust_ID: (nX1 ndarray) refined clusters
    '''
    numPnt = len(indices)  # number of points in cluster
#     dist = distance.squareform(distance.pdist(points[indices, :], 'euclidean'))  # pairwise distance between all the points 
    # nearest neighbor for each point in the pointset 
    triNeighbors = map(functools.partial(TriNeighbors, tri=tri, ind=indices), np.asarray(indices))

    pntSegID = np.zeros((numPnt, 1))  # segment ID for each point 
    segID = 1  # ID of a segment
    sortSeg = np.zeros((numPnt, 1))  # points order in cluster
    
    i = 0 
    pntSegID[i] = -segID
    sortSeg[i] = 1
    order = 1
    while 0 in pntSegID:
        firstN = (triNeighbors[i])[0]  # neighbors from the first triangle ring
        flag = 0
        while flag == 0 and len(firstN) > 0:
            # distance between the point and first tr. ring neighbors
            dist_i = distance.cdist(np.expand_dims(points[indices[i], :], 0), points[firstN, :])
            i_minDist = np.where(dist_i == dist_i.min())[1][0] 
            
            i_indices = np.where(indices == firstN[i_minDist])[0][0]  # index of the closest point in indices list
            if pntSegID[i_indices] > 0 or pntSegID[i_indices] == -segID:  # if min dist is to the point is already in segment
                firstN = np.delete(firstN, i_minDist, 0)
            elif pntSegID[i_indices] < 0 and pntSegID[i_indices] != -segID:  # if a segment is a part of another segment
                order_index = np.where(np.abs(pntSegID) == -pntSegID[i_indices])
                sortSeg[order_index] += sortSeg[i]
                pntSegID[np.where(pntSegID == segID)[0]] = -pntSegID[i_indices]
                pntSegID[np.where(pntSegID == -segID)[0]] = pntSegID[i_indices]
                pntSegID[i_indices] = -pntSegID[i_indices]   
                flag = 1
                i = (np.where(pntSegID == 0))[0][0]  # start to build a new segment
                pntSegID[i] = -segID
                order = 1
                sortSeg[i] = order
            else:
                flag = 1
                pntSegID[i_indices] = segID
                order += 1
                sortSeg[i_indices] = order
                i = i_indices
                    
        if len(firstN) == 0:
            secondN = (triNeighbors[i])[1]
            flag = 0
            while flag == 0 and len(secondN) > 0:
                dist_i = distance.cdist(np.expand_dims(points[indices[i], :], 0), points[secondN, :])
                i_minDist = np.where(dist_i == dist_i.min())[1][0]
                
                i_indices = np.where(indices == secondN[i_minDist])[0][0]  # index of the closest point in indices list
                if pntSegID[i_indices] > 0 or pntSegID[i_indices] == -segID:
                    secondN = np.delete(secondN, i_minDist, 0)
                elif pntSegID[i_indices] < 0 and pntSegID[i_indices] != -segID:  # if a segment is a part of another segment
                    order_index = np.where(np.abs(pntSegID) == -pntSegID[i_indices])
                    sortSeg[order_index] += sortSeg[i]
                    pntSegID[np.where(pntSegID == segID)[0]] = -pntSegID[i_indices]
                    pntSegID[np.where(pntSegID == -segID)[0]] = pntSegID[i_indices]
                    pntSegID[i_indices] = -pntSegID[i_indices]   
                    flag = 1
                    i = (np.where(pntSegID == 0))[0][0]  # start to build a new segment
                    pntSegID[i] = -segID
                    order = 1
                    sortSeg[i] = order
                else:
                    flag = 1
                    pntSegID[i_indices] = segID
                    order += 1
                    sortSeg[i_indices] = order
                    i = i_indices
            
            if len(secondN) == 0:
                segID += 1
                i = (np.where(pntSegID == 0))[0][0]  # start to build a new segment
                order = 1
                pntSegID[i] = -segID
                sortSeg[i] = order
    
    return np.hstack((clust_ID + np.abs(pntSegID), sortSeg))
                
            
@jit(i4[:](double[:, :], i4[:], i4[:], i4[:, :]))    
def Clusters_refining(points, p_ind, clusters, triIndices):
    '''
    cluster refinement
    :Args:
        - points: (nX3 ndarray) all points coordinates
        - p_ind: (nX1 ndarray) indices of points in clusters
        - clusters: (nX1 ndarray) cluster index for each point
        - triIndices: (nX3 ndarray) triangulation indices
    :Returns:
        - clustID_new: (nX2 ndarray) updated clusters' indices and points' order in each cluster
    '''
    points = points.ToNumpy()
    numOfClust = clusters.max()
    clustID_new = np.zeros((len(clusters), 2))
    clID = 0
    for cl in xrange(1, numOfClust + 1):
        cl_i = (np.where(clusters == cl))[0]
        Pnt_ind = p_ind[cl_i]
        clustID_new[cl_i] = CurveSegments(points, Pnt_ind, triIndices, clID)
        clID = clustID_new[:, 0].max()           
        
    return np.int32(clustID_new)


def CurveSmoothing(i, cluster_pnts, clustersID, step):
    '''
    '''
    points = cluster_pnts[clustersID[:, 0] == i]
    numOfPnts = len(points)
    pnt_smoothed = np.zeros((numOfPnts, 3))
    pnt_smoothed[0:step, :] = points[0:step, :]
    for i in xrange(step, numOfPnts - step):
        neighbors = np.vstack((points[i - step:i, :], points[i + 1:i + step + 1, :]))
        dist = distance.cdist(np.expand_dims(points[i, :], 0), neighbors)
        w = 1.0 / dist  # weight function - inverse distance
        dp = (1.0 / (np.sum(w))) * np.sum(w.T * (neighbors - points[i, :]), 0)
        new_pnt = points[i, :] + dp
        pnt_smoothed[i, :] = new_pnt
    pnt_smoothed[numOfPnts - step - 1:numOfPnts, :] = points[numOfPnts - step - 1:numOfPnts, :]

    return pnt_smoothed


def CurvatureInPoint(p1, p2, p3):
    '''
    '''
    norm_3p = np.cross(p1 - p2, p3 - p2)
    norm_3p = norm_3p / np.linalg.norm(norm_3p)
    rotMat = Rotation_Matrix(norm_3p, [0, 0, 1])
    pnts = np.dot(rotMat, np.hstack((np.expand_dims(p1, 1), np.expand_dims(p2, 1), np.expand_dims(p3, 1))))
    p1, p2, p3 = pnts[:, 0], pnts[:, 1], pnts[:, 2]
    
    # triangle's sides
    a = np.sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2)
    b = np.sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2)
    c = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)  
    
    A = 0.5 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))  # The triangle's area including its sign
    return 4 * A / (a * b * c)
    

def Curve_Curvature(i, cluster_pnts, new_clusters, step):
    '''
    curve's curvature
    :Args:
        - cluster_pnts: (nX3 ndarray) clusters points
        - new_clusters: (nX1 ndarray) clusters IDs
        - step: (int) number of points from the middle points to the side ones
    :Returns:
        - curv: segment curvature
    '''
    points = cluster_pnts[new_clusters[:, 0] == i]
    # parametric curve
    dist = np.linalg.norm(points[0:-1, :] - points[1::, :], axis=1)
    curveLength = np.sum(dist)
    cdf = np.hstack((0, np.cumsum(dist)))
    t = cdf / curveLength
    
    stepRangeL = step - 0.05
    stepRangeH = step + 0.05
    if curveLength >= 2 * stepRangeH:
        firstP = np.nonzero((cdf >= stepRangeL) & (cdf <= stepRangeH))[0][0]
        reverseCDF = curveLength - cdf
        lastP = np.nonzero((reverseCDF >= stepRangeL) & (reverseCDF <= stepRangeH))[0][-1]
        
        curv = []
        for ind in xrange(firstP, lastP + 1):
            dist2prev = cdf[ind] - cdf[0:ind]
            dist2next = cdf[ind + 1::] - cdf[ind]
            indPrev = np.nonzero((dist2prev >= stepRangeL) & (dist2prev <= stepRangeH))[0]
            indNext = np.nonzero((dist2next >= stepRangeL) & (dist2next <= stepRangeH))[0]
            if len(indPrev) == 0 or len(indNext) == 0:
                curv.append(0)
            else:
                p1 = points[indPrev[-1], :]
                p3 = points[indNext[0] + ind + 1, :]
                p2 = points[ind, :]
                curv.append(CurvatureInPoint(p1, p2, p3))
            
        curv = np.hstack((np.repeat(curv[0], firstP), curv, np.repeat(curv[-1], len(points) - lastP - 1)))
        
    #     p2 = points[step:-step, :]
    #     p3 = points[2 * step::, :]
    #     p1 = points[0:len(p3), :]
    #     curv = map(CurvatureInPoint, p1, p2, p3)
    #     curv = np.hstack((np.repeat(curv[0], step), curv, np.repeat(curv[-1], step)))
    #     print curv
    #     Visualization.RenderPointSet(PointSet(points), renderFlag='color', color=(0, 1, 0), pointSize=3)
    #     Visualization.Show()
        
        curv = np.abs(curv)  # / np.max(np.abs(curv))
        plt.figure()
        plt.plot(cdf, curv)
        print curv
    else:
        curv = np.zeros((1, len(points)))[0].tolist()
    return curv 


def Curves_resampling(seg_points, segmentID, stepRangeL, stepRangeH):
    '''
    '''
    new_PNT = []
    new_ID = []
    segID = np.unique(segmentID[:, 0])
    for i in segID:
        points = seg_points[segmentID[:, 0] == i]
        # parametric curve
        dist = np.linalg.norm(points[0:-1, :] - points[1::, :], axis=1)
        cdf = np.hstack((0, np.cumsum(dist)))
        
        new_PNT.append(points[0, :])
        k = 1
        j = 0
        
        while j < len(points) - 1:            
            if dist[j] < stepRangeL:
                dist_i = cdf[j + 1::] - cdf[j]
                indNext = np.nonzero((dist_i >= stepRangeL) & (dist_i <= stepRangeH))[0]
                if len(indNext) > 0:
                    k += 1
                    j += (indNext[0] + 1)
                    new_PNT.append(points[j, :])  
                else: j += 1       
            else:
                k += 1
                j += 1
                new_PNT.append(points[j, :])
        new_ID.append(i * np.ones((k, 1), dtype=int))
        
    new_ID = np.asarray(list(itertools.chain.from_iterable(new_ID)))    
    new_PNT = np.asarray(new_PNT)  
    return new_PNT, new_ID

def corr2_coeff(A, B):
    '''
    '''
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(0)[None, :]
    B_mB = B - B.mean(0)[None, :]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(0);
    ssB = (B_mB ** 2).sum(0);

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None, :]))


# @jit(double[:](double[:, :], i4[:, :], i4))        
def Curves_correlation(seg_points1, segmentID1, seg_points2, segmentID2, step):
    '''
    '''
    
    stepRangeL = step - 0.01
    stepRangeH = step + 0.01
     
    firstScanCurves = Curves_resampling(seg_points1, segmentID1, stepRangeL, stepRangeH)
    secondScanCurves = Curves_resampling(seg_points2, segmentID2, stepRangeL, stepRangeH)
     
    firstPnt, firstID = firstScanCurves[0], firstScanCurves[1]
    secondPnt, secondID = secondScanCurves[0], secondScanCurves[1]
     
    id1 = np.unique(firstID)
    id2 = np.unique(secondID)
    for i1 in id1:
        corr = []
        for j1 in id2:
            c1 = firstPnt[np.where(firstID == i1)[0], :]
            c2 = secondPnt[np.where(secondID == j1)[0], :]
            len1, len2 = len(c1), len(c2)
            if len1 < len2:
                temp = c1
                c1 = c2
                c2 = temp
#             corr_i = corr2_coeff(c1, c2)
            corr_x = signal.correlate(c1[:, 0], c2[:, 0])
            corr_y = signal.correlate(c1[:, 1], c2[:, 1])
            corr_z = signal.correlate(c1[:, 2], c2[:, 2])
            corr_i = np.vstack((corr_x, corr_y, corr_z)).T
#             corr1.append(corr_i)   
             
#             Visualization.RenderPointSet(PointSet(c1), renderFlag='color', color=(1, 0, 0), pointSize=3)
#             Visualization.RenderPointSet(PointSet(c2), renderFlag='color', color=(0, 0, 1), pointSize=3)
#              
#             plt.figure()
#             plt.plot(range(len(corr_i)), corr_i)
#             plt.show()
#             Visualization.Show()
          
#         max_corr_i1 = map(max, corr1)
#         ind_MaxCorr1 = max_corr_i1.index(max(max_corr_i1))
#              
#         curve_corr.append(corr1[ind_MaxCorr1])
#         max_corr_ind1[i1] = ind_MaxCorr1
        
     
     
     
#     numOfSeg1 = len(segment_curv[0][0])
#     numOfSeg2 = len(segment_curv[1][0])
#     
# #     corr_C1C2 = lambda c1, c2: np.correlate(c1, c2, "same")
#     curve_corr = []
#     max_corr_ind1 = np.zeros((numOfSeg1, 1))
#     max_corr_ind2 = np.zeros((numOfSeg2, 1))
#     
#     for i1 in xrange(numOfSeg1):
#         corr1 = []
#         for j1 in xrange(numOfSeg2):
# #             seg_i_corr1 = map(functools.partial(corr_C1C2, segment_curv[0][0][i1]), segment_curv[1][0])
#             c1 = segment_curv[0][0][i1]
#             c2 = segment_curv[1][0][j1]
#             len1, len2 = len(c1), len(c2)
# #             if len1 < len2:
# #                 temp = c1
# #                 c1 = c2
# #                 c2 = temp
# #             corr_i = signal.correlate(c1, c2, 'full') / (np.linalg.norm(c1 - np.mean(c1)) * np.linalg.norm(c2 - np.mean(c2)))
# #             corr1.append(corr_i)  
#             Len = len1 + len2 + 1
#             dLen1, dLen2 = Len - len1, Len - len2
#             c2 = (np.hstack((np.expand_dims(c2, 0), np.zeros((1, dLen2)))))[0]
#             c1 = (np.hstack((np.expand_dims(c1, 0), np.zeros((1, dLen1)))))[0]
#             corr_i = np.fft.irfft((np.fft.rfft(c1)) * np.fft.rfft(c2[::-1]))  # / (np.linalg.norm(c1) * np.linalg.norm(c2))
#             corr1.append(corr_i)   
#             
#             plt.figure()
#             plt.plot(range(len(corr_i)), corr_i)
#         
#         max_corr_i1 = map(max, corr1)
#         ind_MaxCorr1 = max_corr_i1.index(max(max_corr_i1))
#             
#         curve_corr.append(corr1[ind_MaxCorr1])
#         max_corr_ind1[i1] = ind_MaxCorr1
#     
# #     for i2 in xrange(numOfSeg2):
# #         corr2 = []
# #         for j2 in xrange(numOfSeg1):
# # #             seg_i_corr1 = map(functools.partial(corr_C1C2, segment_curv[0][0][i1]), segment_curv[1][0])
# #             c2 = segment_curv[1][0][i2]
# #             c1 = segment_curv[0][0][j2]
# #             len1, len2 = len(c1), len(c2)
# #             Len = len1 + len2 + 1
# #             dLen1, dLen2 = Len - len1, Len - len2
# #             c2 = (np.hstack((np.expand_dims(c2, 0), np.zeros((1, dLen2)))))[0]
# #             c1 = (np.hstack((np.expand_dims(c1, 0), np.zeros((1, dLen1)))))[0] 
# #             
# #             corr_i = np.fft.ifft((np.fft.fft(c1)) * np.fft.fft(c2[::-1])) / (np.linalg.norm(c1) * np.linalg.norm(c2))
# #             corr2.append(corr_i)   
# #             
# #             plt.figure()
# #             plt.plot(range(len(corr_i)), corr_i)
# #         
# #         max_corr_i2 = map(max, corr2)
# #         ind_MaxCorr2 = max_corr_i2.index(max(max_corr_i2))
# #             
# #         curve_corr.append(corr2[ind_MaxCorr2])
# #         max_corr_ind2[i2] = ind_MaxCorr2
#     
#     return curve_corr, max_corr_ind1  # , max_corr_ind2


def MapCurve2Cylinder(points1, points2):
    '''
    '''
    len1 = len(points1)
    len2 = len(points2)
    points1 = points1 - np.repeat(np.expand_dims(points1[0, :], 0), len1, axis=0)
    points2 = points2 - np.repeat(np.expand_dims(points2[0, :], 0), len2, axis=0)
    
    if len1 > len2: 
        longCurve = points1
        shortCurve = points2
    else:
        longCurve = points2
        shortCurve = points1
    
    del points1, points2
    
    
    
    
    

#===============================================================================================================================
#                 ==================================== MAIN ==================================== 
#===============================================================================================================================

if __name__ == '__main__':
    
    plt.ion()
    
    print "Loading data..."         
#     (id,x,y,z)
#     dataset for curvature parameters validation: ritual_place_reduced.pts 
    FileNames = ['D:\\Documents\\Pointsets\\habonim_st5_1.pts', 'D:\\Documents\\Pointsets\\habonim_st4_1.pts']  # Bonim_set1.pts', 'D:\\Documents\\Pointsets\\Bonim_set2.pts']  # , 'D:\\Documents\\Pointsets\\habonim_st4.pts']  # Ahziv_big.pts']    points2.pts']  #    ahziv2_set.pts']  #   Bonim_set1.pts', 'D:\\Documents\\Pointsets\\Bonim_set2.pts']  # ']  #  , 'D:\\Documents\\Pointsets\\points1.pts']  # 'D:\\Documents\\Pointsets\\set3_1.pts' pntAhziv.pts']  #
    coeff = 0.3  # radius[m] / scale
    
    segment_curv = [[], []]
    
    curves_points = [[], []]
    curves_ID = [[], []]
    
    for fileName_ind in xrange(2):
    #     ======================================= Data ======================================
    
    # ===== profiling
#         pr = cProfile.Profile()
#         pr.enable()
#         plt.ion()
        
        print "Preparing data..."
        fileName = FileNames[fileName_ind]
        pointSet = []
        IOFactory.ReadPts(fileName, pointSet)
        pointSet = pointSet[0]
        pp = (pointSet.ToNumpy())  # [::3, :]
#         pointSet = PointSet(pp)
        
#         fig_triang = mlab.figure(bgcolor=(0.5, 0.5, 0.5), fgcolor=(1, 1, 1))
#         fig_triang = Visualization.RenderTriangularMesh(triangulation, renderFlag='color', _figure=fig_triang)
#         Visualization.Show()
#         ind = DataSnooping(np.reshape(triangulation.LengthOfAllEdges(), ((3 * triangulation.NumberOfTriangles, 1))), 2)
        
        # ====== KD tree ======
        print "Creating KDtree..."
        tree = cKDTree(pp)
        
#         now = datetime.datetime.now()
#         print datetime.date(now.year, now.month, now.day), datetime.time(now.hour, now.minute, now.second), "Total time:", now - startTime 
        # ====== Ball tree ======
#         tree = BallTree(pp)
#         neighbors = map(functools.partial(NeighborsFactory.GetNeighborsIn3dRange_BallTree, pntSet=pointSet, radius=coeff, tree=tree), pp)
        
# #         fig_data = mlab.figure("Data") 
# #         fig_data = Visualization.RenderPointSet(pointSet, renderFlag='height', pointSize=2, colorBar=1)
#         Visualization.RenderPointSet(pointSet, renderFlag='height', pointSize=2, colorBar=1) 
          
    #     ========================= Points' Normals Computation =============================
#         print "Computing normals..."
#         normals = np.asarray(map(functools.partial(Normal, points=pointSet, coeff=10, pntTree=tree), pp))
#         pointsNormals = NormalsProperty(pointSet, normals)
#         
#         center = np.mean(pp, 0)
#         normC = Normal(center, pointSet, 10)
#         
#         angleN = np.arccos((np.sum(normals * np.repeat(np.expand_dims(normC, 0), normals.shape[0], 0), 1)) / 
#                                        (np.linalg.norm(normals, 2, 1) * np.linalg.norm(normC, 2)))
# #         nRel = np.zeros((1, 3))
# #         indMax = np.where(np.abs(normC) == np.max(np.abs(normC)))[0][0]
# #         nRel[0, indMax] = 1 * np.sign(normC[indMax])
# #         angSign = np.sign(np.sum(nRel * np.cross(normals, np.repeat(np.expand_dims(normC, 0), normals.shape[0], 0), 1), 1))
# #  
# #         angleN[np.where(angSign == -1)] -= 2 * np.pi
# #         angleN *= angSign
#         
# #         n, _, _ = plt.hist(angleN, bins=360, range=[0, 2 * np.pi], facecolor='green')
# #         plt.axis([-0.5, 2 * np.pi, 0, np.max(n) + 5])
#         
#         Visualization.RenderPointSet(pointsNormals, renderFlag="height", pointSize=3)
# #         Visualization.Show()
# #         plt.show()
           
    
         
    #     =========================== Points' Curvature Computation ===========================   
        print "Computing curvature..."
        
#         curv = np.asarray(map(functools.partial(Curvature, coeff=20, points=pointSet), pp))
#         indC = DataSnooping(curv,3)
#         curv = curv[indC]
#         Visualization.RenderPointSet(PointSet(pp[indC, :]), renderFlag='parametericColor',
#                                         color=(curv, np.zeros(curv.size), np.zeros(curv.size)), pointSize=3, colorBar=1) 
        
        now = datetime.datetime.now()
        print datetime.date(now.year, now.month, now.day), datetime.time(now.hour, now.minute, now.second)
        startTime = now
        curv = np.asarray(map(functools.partial(Curvature_FundamentalForm, points=pointSet, rad=coeff), pp))  # , neighbors))
        now = datetime.datetime.now()
        print datetime.date(now.year, now.month, now.day), datetime.time(now.hour, now.minute, now.second), "Total time:", now - startTime
        
        indNotGood = np.where(curv == -999)[0]
        indNotGood = np.unique(indNotGood, return_index=True)[0]
        indGood = np.where(curv != -999)[0]
        indGood = np.unique(indGood, return_index=True)[0]
        curv = np.delete(curv, indNotGood, 0)
        
        pp1 = PointSubSet(pointSet, indGood)  # good points
        del pp
        pp = PointSubSet(pointSet, indNotGood)
        
        spherPoints = SphericalCoordinatesFactory.CartesianToSphericalCoordinates(pp1)
        triangulation = TriangulationFactory.Delaunay2D(spherPoints)
        triangulation.TrimEdgesByLength(0.3)
        
        triangIndices = triangulation.TrianglesIndices
        
#         Visualization.RenderPointSet(PointSet(pp1), renderFlag='color', _figure=fig_data, pointSize=4, color=(0, 0, 0), colorBar=1)
        
        maxCurv = np.max(np.max(np.abs(curv), 0))
        k1 = np.expand_dims(curv[:, 0], 1)
#         k1 = k1 / maxCurv
        k2 = np.expand_dims(curv[:, 1], 1)
#         k2 = k2 / maxCurv
        del curv
        print 'k1 min:  ', np.min(k1)
        print 'k1 max:  ', np.max(k1)
        print 'k2 min:  ', np.min(k2)
        print 'k2 max:  ', np.max(k2)
        
    #=====================================================================================================
    #                        =============== CURVATURES ===============
    #=====================================================================================================
    
        print "Curvature parameters..."
        now = datetime.datetime.now()
        print datetime.date(now.year, now.month, now.day), datetime.time(now.hour, now.minute, now.second)
        startTime = now
        # =========================================== principal curvatures
#         print "Principal curvatures..."
#         indC = DataSnooping(k1,3)
#         k1_t = (k1.T[0])[indC]
# #         fig_k1 = mlab.figure("Principal Curvature: k1")
#         fig_k1 = Visualization.RenderPointSet(PointSubSet(pp1, indC), renderFlag='parametericColor',
#                                         color=(k1_t, np.zeros(k1_t.size), np.zeros(k1_t.size)), pointSize=3, colorBar=1)
#         Visualization.RenderPointSet(pp, renderFlag='color', _figure=fig_k1, color=(0, 0, 0), pointSize=3)
#          
# 
#         indC = DataSnooping(k2,3)
#         k2_t = (k2.T[0])[indC]
# #         fig_k2 = mlab.figure("Principal Curvature: k2")
#         fig_k2 = Visualization.RenderPointSet(PointSubSet(pp1, indC), renderFlag='parametericColor',
#                                         color=(k2_t, np.zeros(k2_t.size), np.zeros(k2_t.size)), pointSize=3, colorBar=1)
#         Visualization.RenderPointSet(pp, renderFlag='color', _figure=fig_k2, color=(0, 0, 0), pointSize=3)
        
        
        # =========================================== mean curvature
        print "Mean curvature..."
        meanCurv = (k1 + k2) / 2
        indC = DataSnooping(meanCurv, 3)
        meanCurv1 = (meanCurv.T[0])[indC]
#         fig_meanCurv = mlab.figure("Mean Curvature")
        fig_meanCurv = Visualization.RenderPointSet(PointSubSet(pp1, indC), renderFlag='parametericColor',
                                        color=(meanCurv1, np.zeros(meanCurv1.size), np.zeros(meanCurv1.size)), pointSize=3, colorBar=1)
        Visualization.RenderPointSet(pp, renderFlag='color', _figure=fig_meanCurv, color=(0, 0, 0), pointSize=3)
        
        
        
        # =========================================== curvadness - a positive number that specifies the amount of curvature
#         print "Curvadness..."
#         curvadness = np.sqrt((k1 ** 2 + k2 ** 2) / 2)
#         indC = DataSnooping(curvadness,3)
#         curvadness = (curvadness.T[0])[indC]
# #         fig_curvadness = mlab.figure("Curvadness")
#         fig_curvadness = Visualization.RenderPointSet(PointSubSet(pp1, indC), renderFlag='parametericColor',
#                                         color=(curvadness, np.zeros(curvadness.size), np.zeros(curvadness.size)), pointSize=3, colorBar=1)
#         Visualization.RenderPointSet(pp, renderFlag='color', _figure=fig_curvadness, color=(0, 0, 0), pointSize=3)
        
        
        # =========================================== shape index
#         print "Shape index..."
#         shapeI = np.zeros(k1.shape)
#         equalZero = np.where(np.abs(k1 - k2) <= 1e-6)[0]
#         difZero = np.where(k1 != k2)[0]
#         if equalZero.size != 0:
#             shapeI[equalZero, :] = 0
#         shapeI[difZero, :] = (1.0 / np.pi) * np.arctan2((k2 + k1)[difZero], (k2 - k1)[difZero])  # 0.5 - (1.0 / np.pi) * np.arctan2(k2 + k1, k2 - k1)
#         indC = DataSnooping(shapeI,3)
#         shapeI = (shapeI.T[0])[indC]
#     
# #         fig_shapeIndex = mlab.figure("Shape Index")
#         fig_shapeIndex = Visualization.RenderPointSet(PointSubSet(pp1, indC), renderFlag='parametericColor',
#                                         color=(shapeI, np.zeros(shapeI.size), np.zeros(shapeI.size)), pointSize=3, colorBar=1)
#         Visualization.RenderPointSet(pp, renderFlag='color', _figure=fig_shapeIndex, color=(0, 0, 0), pointSize=3)
          
    
        
        # =========================================== similarity curvature
        print "Similarity curvature..."
        rgb, similarCurv = Similarity_Curvature(k1, k2)
        rgb = np.asarray(255 * rgb, dtype=np.uint8)
        pntR = PointSet(pp1.ToNumpy(), rgb=rgb)  # np.asarray(255 * (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb)), dtype=np.uint8))
        
# #         fig_simCurv = mlab.figure("Similarity Curvature")
#         fig_simCurv = Visualization.RenderPointSet(pp, renderFlag='color', color=(0, 0, 0), pointSize=3)
#         Visualization.RenderPointSet(pntR, renderFlag='rgb', _figure=fig_simCurv, pointSize=3)
        
        # =========================================== zero crossing
        print "Zero crossing curvatures..."
        i1 = np.where(similarCurv[:, 0] == 0)[0]
        positiveH = i1[np.where(np.sign(similarCurv[i1, 1]) == 1)]
        negativeH = i1[np.where(np.sign(similarCurv[i1, 1]) == -1)]
        
        print "\tCreating KDtree..."
        pntH = pp1.ToNumpy()[np.hstack((positiveH, negativeH)), :]
        tree = cKDTree(pntH)
        
        print "\tZero-crossing..."
        plus2minus = np.asarray(map(functools.partial(ZeroCrossing_Curvature, points=pntH,
                                                      iNegative=np.arange(len(positiveH), len(positiveH) + len(negativeH)),
                                                      meanCurv=meanCurv[np.hstack((positiveH, negativeH)), 0], tree=tree), np.arange(0, len(positiveH))))
        notZero = np.where(plus2minus != 0)
        indC = DataSnooping(plus2minus[notZero], 3)
        zeroCross = ((plus2minus[notZero]).T)[indC]
        
        saddle = positiveH[(notZero[0])[indC]]
         
        zeroCross_pnts = PointSubSet(pp1, saddle)
        fig_ = Visualization.RenderPointSet(zeroCross_pnts, renderFlag='parametericColor',
                                        color=(zeroCross, np.zeros(zeroCross.size), np.zeros(zeroCross.size)),
                                        pointSize=3, colorBar=1)
#         fig_ = Visualization.RenderPointSet(PointSubSet(pp1, saddle), renderFlag='color', color=(1, 0, 0), pointSize=3)
        
        fig_ = Visualization.RenderPointSet(pp, renderFlag='color', color=(0, 0, 0), _figure=fig_, pointSize=3)
        fig_ = Visualization.RenderPointSet(PointSubSet(pp1, np.where(similarCurv[:, 1] == 0)[0]), renderFlag='color', color=(0, 0, 0),
                                             _figure=fig_, pointSize=3)
        fig_ = Visualization.RenderPointSet(PointSubSet(pp1, negativeH), renderFlag='color', color=(0, 0, 0), _figure=fig_, pointSize=3)
        Visualization.RenderPointSet(PointSubSet(pp1, positiveH[np.where(plus2minus == 0)]), renderFlag='color', color=(0, 0, 0), _figure=fig_, pointSize=3)
        
        fig_triang = mlab.figure(bgcolor=(0.5, 0.5, 0.5), fgcolor=(1, 1, 1))
        fig_triang = Visualization.RenderPointSet(zeroCross_pnts, renderFlag='parametericColor', _figure=fig_triang,
                                        color=(zeroCross, np.zeros(zeroCross.size), np.zeros(zeroCross.size)),
                                        pointSize=4, colorBar=1)
        Visualization.RenderTriangularMesh(triangulation, renderFlag='color', _figure=fig_triang)
        
#         triangulation.TrimEdgesByLength(0.2)
        
        now = datetime.datetime.now()
        print datetime.date(now.year, now.month, now.day), datetime.time(now.hour, now.minute, now.second), "Total time:", now - startTime
        
        # ==== end profiling     
#         pr.disable()
#         s = StringIO.StringIO()
#         sortby = 'cumulative'
#         ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#         ps.print_stats(20)
#         print s.getvalue()
   
     
        # ===== clusters =====
        clusters = (CurveClusters(i=0, points=zeroCross_pnts, maxClust=np.sqrt(zeroCross_pnts.Size / 2)))
        numOfClust = clusters.max()
        # cluster refinement
        new_clusters = Clusters_refining(pp1, saddle, clusters, triangIndices)
        # delete short segments
        cluster_pnts = pp1.ToNumpy()[saddle, :]
        
        clSort = np.argsort(new_clusters[:, 0])
        new_clusters = new_clusters[clSort]
        cluster_pnts = cluster_pnts[clSort]
        cl_ID = np.arange(1, new_clusters[:, 0].max() + 1)
        
        for label in xrange(1, new_clusters[:, 0].max() + 1):
            i = np.where(new_clusters[:, 0] == label)[0]
            if len(i) < 40:
                cluster_pnts = np.delete(cluster_pnts, i, 0)
                new_clusters = np.delete(new_clusters, i, 0)
                cl_ID = np.delete(cl_ID, np.where(cl_ID == label), 0)
            else: 
                orderSort = np.argsort(new_clusters[i, 1])
                new_clusters[i, :] = (new_clusters[i, :])[orderSort]
                cluster_pnts[i, :] = (cluster_pnts[i, :])[orderSort]
        fig_cl = Visualization.RenderPointSet(PointSet(cluster_pnts), renderFlag='parametericColor',
                                color=(new_clusters, np.zeros(new_clusters.size), np.zeros(new_clusters.size)), pointSize=4, colorMap='Paired', colorBar=1)
        Visualization.RenderTriangularMesh(triangulation, renderFlag='color', _figure=fig_cl)
                
        # ============= curves smoothing
        newCoords = map(functools.partial(CurveSmoothing, cluster_pnts=cluster_pnts, clustersID=new_clusters, step=2), cl_ID)
        newCoords = np.asarray(list(itertools.chain.from_iterable(newCoords)))
        fig_cl_smooth = Visualization.RenderPointSet(PointSet(newCoords), renderFlag='parametericColor',
                                color=(new_clusters, np.zeros(new_clusters.size), np.zeros(new_clusters.size)), pointSize=4, colorMap='Paired', colorBar=1)
        Visualization.RenderTriangularMesh(triangulation, renderFlag='color', _figure=fig_cl_smooth)
        
        curves_points[fileName_ind].append(newCoords)
        curves_ID[fileName_ind].append(new_clusters)
        
        # ============= curves curvature
        curv_val_out = map(functools.partial(Curve_Curvature, cluster_pnts=newCoords, new_clusters=new_clusters, step=0.2), cl_ID)
        segment_curv[fileName_ind].append(curv_val_out)
        curv_val = np.asarray(list(itertools.chain.from_iterable(curv_val_out)))
#         print curv_val
          
        fig_cl_new = Visualization.RenderPointSet(PointSet(cluster_pnts), renderFlag='parametericColor',
                                color=(curv_val, np.zeros(curv_val.size), np.zeros(curv_val.size)), pointSize=3, colorMap='Paired', colorBar=1)
        Visualization.RenderTriangularMesh(triangulation, renderFlag='color', _figure=fig_cl_new)
        
        
        
        fig2Close = mlab.figure(bgcolor=(0.5, 0.5, 0.5), fgcolor=(1, 1, 1))
#         # az,el,dist
#         spherPoints = (SphericalCoordinatesFactory.CartesianToSphericalCoordinates(pointSet)).ToNumpy()
#         spherPntSet = PointSet(spherPoints)
#         
#         Visualization.RenderPointSet(spherPntSet, renderFlag='height', pointSize=3)
    #       
    #     print "Gaussian smoothing..."
    #     
    # #     for i in range(10, 60, 10):
    # #         d_new = np.asarray(map(functools.partial(GaussianSmooth, points=spherPntSet, res=0.1, coeff=i), spherPoints))
    # #         spher_new = np.hstack((spherPoints[:, 0:2], np.expand_dims(d_new, 1)))
    # #         new_points = SphericalCoordinatesFactory.SphericalToCartesianCoordinates(spher_new)
    # # #         Visualization.RenderPointSet(new_points, renderFlag='height', color=(0, 1, 0), pointSize=2)
    # #         smoothedPoints = SphericalCoordinatesFactory.SphericalToCartesianCoordinates(spher_new)
    # #         for j in range(10, 60, 20):
    # #             curv = np.asarray(map(functools.partial(Curvature, points= smoothedPoints.ToNumpy(), coeff=j),  smoothedPoints.ToNumpy()))
    # #             Visualization.RenderPointSet(smoothedPoints, renderFlag='parametericColor',
    # #                                          color=(curv, np.zeros(curv.size), np.zeros(curv.size)), pointSize=2)  
    #             
    #     d_new = np.asarray(map(functools.partial(GaussianSmooth, points=spherPntSet, res=0.1, coeff=10), spherPoints))
    #     spher_new = np.hstack((spherPoints[:, 0:2], np.expand_dims(d_new, 1)))
    #     new_points = SphericalCoordinatesFactory.SphericalToCartesianCoordinates(spher_new)
    #     smoothedPoints = SphericalCoordinatesFactory.SphericalToCartesianCoordinates(spher_new)
    #     
    # #     surf_Var = []
    # #     print "Calculating Surface Variation..."
    # #     for i in range(10, 60, 10):
    # #         var = np.asarray(map(functools.partial(SurfVariation, points=smoothedPoints, coeff=i), smoothedPoints.ToNumpy()))
    # #         surf_Var.append(var)
    # # #         Visualization.RenderPointSet(smoothedPoints, renderFlag='parametericColor',
    # # #                                      color=(var, np.zeros(var.size), np.zeros(var.size)), pointSize=2) 
    # #     var_ = np.max(np.asarray(surf_Var), 0)
    # #     Visualization.RenderPointSet(smoothedPoints, renderFlag='parametericColor',
    # #                                     color=(var_, np.zeros(var_.size), np.zeros(var_.size)), pointSize=2) 
    # 
    #     curvature = []
    #     print "Calculating Curvature..."
    #     for i in range(10, 60, 10):
    #         curv = np.asarray(map(functools.partial(Curvature, points=smoothedPoints, coeff=i), smoothedPoints.ToNumpy()))
    #         curvature.append(curv)
    # #         Visualization.RenderPointSet(smoothedPoints, renderFlag='parametericColor',
    # #                                      color=(curv, np.zeros(curv.size), np.zeros(curv.size)), pointSize=2)       
    # 
    #     curv_ = np.max(np.asarray(curvature), 0)
    #     Visualization.RenderPointSet(smoothedPoints, renderFlag='parametericColor',
    #                                     color=(curv_, np.zeros(curv_.size), np.zeros(curv_.size)), pointSize=2) 
    
    
    # ============ curves correlation =============  
#     corr = Curves_correlation(curves_points[0][0], curves_ID[0][0], curves_points[1][0], curves_ID[1][0], 0.05)   
#     curve_corr, max_corr_ind1 = Curves_correlation(segment_curv)
#     print curve_corr, '\n', max_corr_ind1  # , '\n', max_corr_ind2
    Visualization.Show()
    plt.show()
    
