'''
Created on Nov 26, 2014

@author: Vera
'''

import glob
import os

import numpy as np
from scipy.linalg import inv as invSp

from IOFactory import IOFactory
from PointSet import PointSet
from VisualizationVTK import VisualizationVTK


def Fit_Plane( points, ( n, d ), f, f1 ):
    '''
    Plane fitting
     
    :Args:
        - points: points coordinates (x,y,z)
        - n: normal
        - d: parameter d of plane equation
        - f: output file
     
    :Returns:
        - x,y,x0: corrected values and fitted coefficients
     
    '''
    
    x = np.array( [points[:, 0]] ).T
    y = np.array( [points[:, 1]] ).T
    z = np.array( [points[:, 2]] ).T
    m = x.shape[0]
    
    x0 = np.vstack( ( n[0], n[1], n[2], d ) )

    A = np.hstack( ( x, y, z, -np.ones( z.shape ) ) )
    B = np.hstack( ( np.diag( n[0] * np.ones( ( 1, z.size ) )[0] ),
                     np.diag( n[1] * np.ones( ( 1, z.size ) )[0] ),
                     np.diag( n[2] * np.ones( ( 1, z.size ) )[0] ) ) )
     
    flag = 0
    lamb = 1e-3
    V1 = np.zeros( ( x.size, 1 ) )
    it = 0
     
    while flag == 0:
        it += 1
        w = x0[0] * x + x0[1] * y + x0[2] * z - x0[3]
        M = np.dot( B, B.T )
        N = np.dot( np.dot( A.T, invSp( M ) ), A )
        U = np.dot( np.dot( A.T, invSp( M ) ), w )
        diagN = np.diag( np.diag( N ) )
         
        dx = np.dot( -invSp( N + lamb * diagN ), U )
        x0 += dx 
        V = np.dot( np.dot( -B.T, invSp( M ) ), ( w + np.dot( A, dx ) ) )
 
        if ( it != 1 ):
            if ( np.abs( np.sum( V ) ) < np.abs( np.sum( V1 ) ) ):
                lamb /= 10
            else:
                if ( np.abs( np.sum( V ) ) > np.abs( np.sum( V1 ) ) ):
                    lamb *= 10
 
          
        V1 = V
     
        if ( np.max( np.max( np.abs( V1 - V ) ) ) < 1e-4 ):
            sigma2 = np.dot( V.T, V ) / ( m - 4 )
            acc = np.sqrt( np.diag( sigma2 * np.linalg.inv( N ) ) )
            RMSE = np.sqrt( np.dot( V.T, V ) / ( m ) )
            flag = 1
         
        B = np.hstack( ( np.diag( x0[0] * np.ones( ( 1, z.size ) )[0] ), np.diag( x0[1] * np.ones( ( 1, z.size ) )[0] ), np.diag( x0[2] * np.ones( ( 1, z.size ) )[0] ) ) )
    
    n = x0.T[0][0:3] / np.linalg.norm( x0[0:3], 2 )
    lenV = len( V )
    v_sort = np.sort( np.abs( V ) )
    five_per = np.int32( np.ceil( lenV * 0.05 ) )
    ten_per = np.int32( np.ceil( lenV * 0.10 ) )

    
    f1.write( 'Normal: ' + str( x0.T[0][0:3] / np.linalg.norm( x0[0:3], 2 ) ) + '\n' ) 
#     f1.write( 'Accuracy: ' + str( acc[0:3] ) + '\n' )        
    f1.write( 'RMSE [m]: ' + str( RMSE[0] ) + '\n' )
    f1.write( 'Absolute Error Mean: ' + str( np.mean( np.abs( V ) ) ) + '\n' )
    f1.write( 'Maximum Absolute Error: ' + str( np.max( np.abs( V ) ) ) + '\n' )
    f1.write( '5% Absolute Error: ' + str( np.mean( v_sort[lenV - five_per::] ) ) + '\n' )
    f1.write( '10% Absolute Error: ' + str( np.mean( v_sort[lenV - ten_per::] ) ) + '\n' )
    
    
    f.write( str( n[0] ) + '\t' + str( n[1] ) + '\t' + str( n[2] ) + '\t' ) 
#     f.write( str( acc[0] ) + '\t' + str( acc[1] ) + '\t' + str( acc[2] ) + '\t' )        
    f.write( str( RMSE[0, 0] ) + '\t' )
    f.write( str( np.mean( np.abs( V ) ) ) + '\t' )
    f.write( str( np.max( np.abs( V ) ) ) + '\t' )
    f.write( str( np.mean( v_sort[lenV - five_per::] ) ) + '\t' )
    f.write( str( np.mean( v_sort[lenV - ten_per::] ) ) + '\n' )
     
    return x + V[0:V.size / 3], y + V[V.size / 3:2 * V.size / 3], z + V[2 * V.size / 3::], x0
    
    
def PlaneMatch( points, f = None, f1 = None ):
    '''
    Initial guess for the parameters
     
    :Args:
        - points: points coordinates
     
    :Returns:
        - norm, d: initial guess for the parameters 
    '''
    
    covCell = np.cov( points.T )  # covariance matrix of pointset
    eigVal, eigVec = np.linalg.eig( covCell )  # eigVal and eigVec of the covariance matrix 
    norm_ind = np.where( eigVal == np.min( eigVal ) )
    norm = ( eigVec[:, norm_ind] ).T[0][0]
    d = np.sum( norm.T * np.mean( points, 0 ) )
    if np.abs( d ) < 1e-6: d = 0
#     print norm, d
    
    V = norm.T.dot( points.T ) - d
    pnt_new = points - np.expand_dims( V, 1 ) * np.repeat( np.expand_dims( norm, 0 ), len( points ), axis = 0 )
    
    lenV = len( V )
    RMSE = np.sqrt( np.dot( V.T, V ) / ( lenV ) )
     
    v_sort = np.sort( np.abs( V ) )
    five_per = np.int32( np.ceil( lenV * 0.05 ) )
    ten_per = np.int32( np.ceil( lenV * 0.10 ) )
    
    if f != None and f1 != None:
        f1.write( 'Normal: ' + str( norm / np.linalg.norm( norm, 2 ) ) + '\n' )     
        f1.write( 'RMSE [m]: ' + str( RMSE ) + '\n' )
        f1.write( 'Absolute Error Mean: ' + str( np.mean( np.abs( V ) ) ) + '\n' )
        f1.write( 'Maximum Absolute Error: ' + str( np.max( np.abs( V ) ) ) + '\n' )
        f1.write( '5% Absolute Error: ' + str( np.mean( v_sort[lenV - five_per::] ) ) + '\n' )
        f1.write( '10% Absolute Error: ' + str( np.mean( v_sort[lenV - ten_per::] ) ) + '\n' )
          
          
        f.write( str( norm[0] ) + '\t' + str( norm[1] ) + '\t' + str( norm[2] ) + '\t' )      
        f.write( str( RMSE ) + '\t' )
        f.write( str( np.mean( np.abs( V ) ) ) + '\t' )
        f.write( str( np.max( np.abs( V ) ) ) + '\t' )
        f.write( str( np.mean( v_sort[lenV - five_per::] ) ) + '\t' )
        f.write( str( np.mean( v_sort[lenV - ten_per::] ) ) + '\n' )
    return norm, d, pnt_new


if __name__ == '__main__':
    
    '''
    f(x,y,z)=nx*x + ny*y + nz*z + d
    '''
#     np.set_printoptions( threshold = 'nan' )  # display all the values in the array
    
    print "Loading data..."         
#     (id,x,y,z)
    dir = r'D:\Qsync\Maatz\MobileIlaniyya\IlaniyyaHouses\ReumaNeighbourhood' 
    os.chdir( dir )
    fileNames = glob.glob( "*.pts" )  # take all files of a specific format
    
#     f = open( dir + '\\Results\\' + 'output.txt', 'w' )
    for fileName in fileNames:
        print "Preparing data..."
        pointSet = []
        IOFactory.ReadPts( fileName, pointSet )
        if len( pointSet ) > 1:
            pointSet = IOFactory.MergePntList( pointSet )
        else:
            pointSet = pointSet[0]
#         f, f1 = None, None
#         f1 = open( dir + '\\Results\\' + fileName[0:-4] + '_output.txt', 'w' )
#         f.write( fileName[0:-4] + '\t' )
        print 'Num. of points: ', pointSet.Size
        points = pointSet.ToNumpy()
        points = points - np.mean( points, 0 )
        coeff0 = PlaneMatch( points )  # , f, f1 )  # Initial guess for the parameters
        
        # draw data points
        pointSet = PointSet( points )
        fig = VisualizationVTK.RenderPointSet(pointSet, renderFlag='color', color=(1, 0, 0), pointSize=2)
           
        print "Parameters' calculation..."
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
         
        # draw fitted points
        
        pointSet_new = PointSet( coeff0[2] )
#         pointSet_new = PointSet( np.hstack( ( x_ney, y_new, z_new ) ) )
        fig = VisualizationVTK.RenderPointSet(pointSet_new, renderFlag='color', _figure=fig, color=(0, 0, 1),
                                              pointSize=2)
        
#         f1.close()
#     f.close()
    VisualizationVTK.Show()
