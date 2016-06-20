'''
Created on Nov 2, 2014

@author: Vera
'''

import numpy as np
from scipy.spatial import cKDTree
from Registration import Registration
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def MovingDirection(pntSet, b, p_gm, s_pi, max_dim):
    '''
    check weather the increase or decrease of b can better fit the optimization function
    :Args:
        - pntSet: ndarray pointset
        - b: parameter
        - p_gm: 1x3 array geometric median
        - s_pi: points' density in ball neighborhood
        - max_dim: max dimension of the pointset
    
    :Returns: 
        - new b
        - new w: points' weights  
    '''
    b_neg = b - 0.01
    w_neg = WeightCalc(pntSet, b_neg, p_gm, s_pi, max_dim)
    gm_neg = np.sum(w_neg * np.asarray([np.linalg.norm((pntSet - p_gm), 2, 1) ** 2]).T)
    
    b_pos = b + 0.01
    w_pos = WeightCalc(pntSet, b_pos, p_gm, s_pi, max_dim)
    gm_pos = np.sum(w_pos * np.asarray([np.linalg.norm((pntSet - p_gm), 2, 1) ** 2]).T)
    
    if gm_neg < gm_pos:
        return b_neg, w_neg
    else:
        return b_pos, w_pos


def WeightCalc(pntSet, b, p_gm, s_pi, max_dim):
    '''
    calculate points' weights as a function of Gaussian distance and density in point's neighborhood
    :Args:
        - pntSet: ndarray pointset
        - b: parameter
        - p_gm: 1x3 array geometric median
        - s_pi: points' density in ball neighborhood
        - max_dim: max dimension of the pointset
    
    :Returns: 
        - new w: points' weights
    '''
    s2 = (max_dim / 6) ** 2
    G = np.exp(-np.linalg.norm((pntSet - p_gm), 2, 1) ** 2 / (2.0 * s2)) / (np.sqrt(s2 * 2.0 * np.pi))
    w = np.asarray([G]).T / (s_pi ** b)
    
    return w


def Weighting(pntSet, s_pi):
    '''
    compute geometric median and points' weights
    :Args:
        - pntSet: ndarray pointset
    :Returns:
        - w: weights
        - p_gm: geometric median of the points
    ''' 
    
    max_dim = np.max(np.abs(np.max(pntSet, 0) - np.min(pntSet, 0)))  # pointset dimensions
    psSize = pntSet.shape[0]
#     initial values  
    b = 1.0
    p_gm = np.asarray([np.sum(pntSet, 0)]) / psSize
    
#     plt.ion()
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(pntSet[:, 0], pntSet[:, 1], pntSet[:, 2], c='b', marker='.', s=5)
#     ax.scatter(p_gm[0, 0], p_gm[0, 1], p_gm[0, 2], c='r', marker='x', s=200)
    

#     initial weight computation     
    w = WeightCalc(pntSet, b, p_gm, s_pi, max_dim)
    
    flag = 0
    while flag == 0:
        print p_gm
        pgm0 = p_gm
        b, w = MovingDirection(pntSet, b, p_gm, s_pi, max_dim)
        p_gm = np.asarray([np.sum(np.hstack((w, w, w)) * pntSet, 0) / (np.sum(w))])
        
        print np.linalg.norm(p_gm - pgm0, 2, 1)
        if np.linalg.norm(p_gm - pgm0, 2, 1) < 4e-4:
            flag = 1
#             fig = plt.figure()
#             ax = Axes3D(fig)
#             ax.scatter(pntSet[:, 0], pntSet[:, 1], pntSet[:, 2], c='b', marker='.', s=5)
#             ax.scatter(p_gm[0, 0], p_gm[0, 1], p_gm[0, 2], c='r', marker='x', s=200)
#             
#             plt.show()
#             plt.show(block=True)  # show all figures

    return w, p_gm 
    

def EigenFeatures(pntSet):
    '''
    compute eigenvalues and eigenvectors
    :Args:
        - pntSet: pointset
    :Returns: 
        - eigVal: eigenvalues
        - eigVec: eigenvectors
    '''
    #     points density within the neighborhood ball

    rad = 0.5 * np.max(np.abs(np.max(pntSet, 0) - np.min(pntSet, 0)))
    psSize = pntSet.shape[0]
    neighbor = np.zeros((psSize, 1))
    tree = cKDTree(pntSet)
    i = 0
    for point in pntSet:
        l = tree.query(point, psSize, p=2, distance_upper_bound=rad)
        neighbor[i] = np.where(l[0] != np.inf)[0].size - 1
        i += 1
    s_pi = neighbor / psSize
    no_neighbor = np.where(s_pi == 0)[0]
    if no_neighbor.size != 0:
        s_pi = np.array([np.delete(s_pi, no_neighbor)]).T
        pntSet = np.delete(pntSet, no_neighbor, 0)
    
    w, p_gm = Weighting(pntSet, s_pi)
    Pi_gm = pntSet - p_gm
    C = np.dot((np.hstack((w, w, w)) * Pi_gm).T, Pi_gm) / np.sum(w)  # weighted covariance matrix of pointset
    eigVal, eigVec = np.linalg.eig(C)
    
    covCell = np.cov(pntSet.T)  # covariance matrix of pointset
    eigVal1, eigVec1 = np.linalg.eig(covCell)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pntSet[:, 0], pntSet[:, 1], pntSet[:, 2], c='b', marker='.', s=5)
    ax.scatter(p_gm[0, 0], p_gm[0, 1], p_gm[0, 2], c='r', marker='x', s=200)
    for i in range(3):
        ax.plot([p_gm[0, 0] , p_gm[0, 0] + eigVec[0, i]], [p_gm[0, 1] , p_gm[0, 1] + eigVec[1, i]],
                [p_gm[0, 2] , p_gm[0, 2] + eigVec[2, i]], c='r')
        ax.plot([p_gm[0, 0] , p_gm[0, 0] + eigVec1[0, i]], [p_gm[0, 1] , p_gm[0, 1] + eigVec1[1, i]],
                [p_gm[0, 2] , p_gm[0, 2] + eigVec1[2, i]], c='b')
    
#     plt.show()
    plt.show(block=True)
    
    return eigVal, eigVec
    


if __name__ == '__main__':

    # Load pointset
    print "Loading data..."         
    fileName = 'D:\\Documents\\Pointsets\\small_sample.pts'   
    
    print "Preparing data..."
    points = Registration.PointSet2Array(fileName)
    points = points[::2, :]
    D = np.abs(np.max(points, 0) - np.min(points, 0))
    print D
    
    print EigenFeatures(points)

