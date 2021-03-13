'''
Created on Nov 26, 2014

@author: Vera

Bug fixes and update to Python3 on Dec 1, 2019
@author: Reuma
'''

import glob
import os

import numpy as np
from scipy.linalg import inv as invSp

from IOFactory import IOFactory
from PointSet import PointSet
# from VisualizationVTK import VisualizationVTK

def fit_bi_quadratic_LS(pointset):
    """
    Fitting a bi-quadratic surface via least squares

    :param pointset:  points coordinates (x,y,z)

    :type pointset: PointSet, BaseData.BaseData

    :return:  x0: fitted coefficients, RMSE: sigma a-postriori squared, v

    :rtype: tuple

    Bi-quadratic surface is defined by:

    .. math::
        z = ax^2 + by^2 + cxy + dx + ey + f

    """
    # 1. model
    n = pointset.Size

    f = np.ones((n, ))

    # 2. move to centroid
    xyz_c = pointset.ToNumpy() - pointset.ToNumpy().mean(axis=0)
    A = np.vstack((xyz_c[:,0]**2, xyz_c[:,1]**2, xyz_c[:,0]* xyz_c[:,1], xyz_c[:,0], xyz_c[:,1], f)).T

    # 2. solution
    N = A.T.dot(A)
    u = A.T.dot(pointset.Z)
    x0 = np.linalg.solve(N, u)

    v = A.dot(x0) - pointset.Z
    sig2 = v.T.dot(v) / (n-6)

    return x0, np.sqrt(sig2), v


def fit_plane_GH(pointset, n, d, eps=1e-4, max_it=10000, path_filename=None, path_metadata=None):
    '''
    Plane fitting using the Least Square adjustment via Gauss-Helmert Model (condition equation with parameters)
     
    :param pointset: points coordinates (x,y,z)
    :param n: normal
    :param d: parameter d of plane equation
    :param eps: epsilon for convergence
    :param max_it: maximum iterations allowed. :TODO: make an option where the number of iterations is unlimited
    :param path_filename: output file. Default: None
    :param path_metadata: output metadata file. Default: None

    :type pointset: PointSet, BaseData.BaseData
    :type n: np.array
    :type d: float
    :type eps: float
    :type max_it: int
    :type path_filename: str
    :type path_metadata: str

    :return: x0: fitted coefficients, RMSE: sigma a-postriori squared, v

    :rtype: tuple
     
    '''
    # 1. model
    m = pointset.Size
    x0 = np.hstack((n, d))
    A = np.vstack((pointset.X, pointset.Y, pointset.Z, -np.ones(m))).T

    # 2. general initializations
    sigma2= 1
    lamb = 1e-3
    v1 = np.zeros((m, 1))
    it = 0
    v = 0
    N = 0

    while sigma2 > eps:
        it += 1
        if it >= max_it:
            break
        B = np.hstack((np.diag(x0[0] * np.ones((1, m))[0]),
                       np.diag(x0[1] * np.ones((1, m))[0]),
                       np.diag(x0[2] * np.ones((1, m))[0])))
        w = x0[0] * pointset.X + x0[1] * pointset.Y + x0[2] * pointset.Z - x0[3]
        M = np.dot(B, B.T)
        N = np.dot(np.dot(A.T, invSp(M)), A)
        U = np.dot(np.dot(A.T, invSp(M)), w)
        diagN = np.diag(np.diag(N))

        dx = np.dot(-invSp(N + lamb * diagN), U)
        x0 += dx
        v = np.dot(np.dot(-B.T, invSp(M)), (w + np.dot(A, dx)))

        if (it != 1):
            if (np.abs(np.sum(v)) < np.abs(np.sum(v1))):
                lamb /= 10
            else:
                if (np.abs(np.sum(v)) > np.abs(np.sum(v1))):
                    lamb *= 10
        v1 = v

        sigma2 = np.dot(v.T, v) / (m - 4)
    acc = np.sqrt(np.diag(sigma2 * np.linalg.inv(N)))
    RMSE = np.sqrt(sigma2)
    n = x0[:3] / np.linalg.norm(x0[0:3], 2)
    lenV = len(v)
    v_sort = np.sort(np.abs(v))
    five_per = np.int32(np.ceil(m * 0.05))
    ten_per = np.int32(np.ceil(m * 0.10))

    if path_metadata is not None:
        with open(path_metadata, 'w') as f_metadata:
            f_metadata.write('Normal: ' + str(x0.T[0][0:3] / np.linalg.norm(x0[0:3], 2)) + '\n')
            #     f1.write( 'Accuracy: ' + str( acc[0:3] ) + '\n' )
            f_metadata.write('RMSE [m]: ' + str(RMSE[0]) + '\n')
            f_metadata.write('Absolute Error Mean: ' + str(np.mean(np.abs(v))) + '\n')
            f_metadata.write('Maximum Absolute Error: ' + str(np.max(np.abs(v))) + '\n')
            f_metadata.write('5% Absolute Error: ' + str(np.mean(v_sort[m - five_per::])) + '\n')
            f_metadata.write('10% Absolute Error: ' + str(np.mean(v_sort[m - ten_per::])) + '\n')

    if path_filename is not None:
        with open(path_filename, 'w') as file:
            file.write(str(n[0]) + '\t' + str(n[1]) + '\t' + str(n[2]) + '\t')
            #     f.write( str( acc[0] ) + '\t' + str( acc[1] ) + '\t' + str( acc[2] ) + '\t' )
            file.write(str(RMSE[0, 0]) + '\t')
            file.write(str(np.mean(np.abs(v))) + '\t')
            file.write(str(np.max(np.abs(v))) + '\t')
            file.write(str(np.mean(v_sort[lenV - five_per::])) + '\t')
            file.write(str(np.mean(v_sort[lenV - ten_per::])) + '\n')

    return x0, RMSE, v

def fit_plane_PCA(pointset, rmse=False):
    r"""
    Initial guess for the parameters using PCA

    :param pointset: points coordinates
    :param rmse: flag to return RMSE. Default: false

    :type pointset: PointSet, BaseData.BaseData

    :type rmse: bool

    :return: norm, d: initial guess for the parameters
    :rtype: tuple

    .. math::
        {\bf C} = \sum \left(p_i - c\right)^T\left(p_i - c\right)

    with :math:`c` the centroid. The normal to the plane is the eigenvector of the smallest eigenvalue
    """

    points = pointset.ToNumpy() - np.mean(pointset.ToNumpy(), 0)

    covCell = (points.T.dot(points)) /pointset.Size # covariance matrix of pointset
    eigVal, eigVec = np.linalg.eig(covCell)  # eigVal and eigVec of the covariance matrix
    norm_ind = np.where(eigVal == np.min(eigVal))

    norm = (eigVec[:, norm_ind]).T[0][0]

    d = np.sum(norm.T * np.mean(pointset.ToNumpy(), 0)) # plane constant

    v = norm.T.dot(pointset.ToNumpy().T) - d
    # pnt_new = points - np.expand_dims(v, 1) * np.repeat(np.expand_dims(norm, 0), len(points), axis=0)

    lenV = len(v)
    RMSE = np.sqrt(np.dot(v.T, v) / (lenV))
    if rmse:
        return (norm, d), RMSE, v
    else:
        return norm, d


if __name__ == '__main__':

    '''
    f(x,y,z)=nx*x + ny*y + nz*z + d
    '''
    #     np.set_printoptions( threshold = 'nan' )  # display all the values in the array

    print("Loading data...")
    #     (id,x,y,z)
    dir = r'D:\Qsync\Maatz\MobileIlaniyya\IlaniyyaHouses\ReumaNeighbourhood'
    os.chdir(dir)
    fileNames = glob.glob("*.pts")  # take all files of a specific format

    #     f = open( dir + '\\Results\\' + 'output.txt', 'w' )
    for fileName in fileNames:
        print("Preparing data...")
        pointSet = []
        pointSet = IOFactory.ReadPts(fileName, pointSet, merge=True)

        print('Num. of points: ', pointSet.Size)
        points = pointSet.ToNumpy()

        coeff0 = fit_plane_PCA(points)  # , f, f1 )  # Initial guess for the parameters

        # draw data points
        from VisualizationO3D import VisualizationO3D
        pointSet = PointSet(points)
        vis = VisualizationO3D()
        vis.visualize_pointset(pointSet)

        print("Parameters' calculation...")
        # least square with Levenberg-Marquardt algorithm
        #         x_ney, y_new, z_new, coeff = Fit_Plane( points, coeff0, f, f1 )

        #         n1 = np.expand_dims( coeff0[0], 1 )
        #         n2 = coeff[0:3]
        #         delta_N = np.rad2deg( np.arccos( np.sum( n1 * n2 ) / ( np.linalg.norm( n1, 2 ) * np.linalg.norm( n2, 2 ) ) ) )
        #         print '\nThe angle between normals: ', delta_N, ' deg\n'

        #         # incidence angle
        #         norm = np.repeat( coeff[0:3].T, points.shape[0], 0 )
        #         incidence_angles = np.rad2deg( np.arccos( np.sum( points * norm, 1 ) / ( np.linalg.norm( points, 2, 1 ) * np.linalg.norm( norm, 2, 1 ) ) ) )
        #         f1.write( 'Incidence Angles [deg]: ' + '\n' + str( incidence_angles ) )

    #     # draw fitted points
    #
    #     pointSet_new = PointSet(coeff0[2])
    #     #         pointSet_new = PointSet( np.hstack( ( x_ney, y_new, z_new ) ) )
    #     fig = VisualizationVTK.RenderPointSet(pointSet_new, renderFlag='color', _figure=fig, color=(0, 0, 1),
    #                                           pointSize=2)
    #
    # #         f1.close()
    # #     f.close()
    # VisualizationVTK.Show()
