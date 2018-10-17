'''
Created on Nov 21, 2014

@author: Vera
'''

import glob
import os

import numpy as np
import scipy.spatial.distance as distance
from scipy.linalg import inv as invSp

from IOFactory import IOFactory
from PointSet import PointSet
from VisualizationVTK import VisualizationVTK


def Rotation_Matrix( a1, a2 ):
    '''
    compute rotation matrix from vector a1 to a2
    :Args:
        - a1,a2: vectors
    :Returns:
        - R: rotation matrix
    '''
    v = np.cross( a1, a2 )
    c = np.sum( a1 * a2 )
    h = ( 1 - c ) / ( 1 - c ** 2 )
    
    R = np.array( [[c + h * v[0] ** 2, h * v[0] * v[1] - v[2], h * v[0] * v[2] + v[1]],
                 [h * v[0] * v[1] + v[2], c + h * v[1] ** 2, h * v[1] * v[2] - v[0]],
                 [h * v[0] * v[2] - v[1], h * v[1] * v[2] + v[0], c + h * v[2] ** 2]] )

    return R


def Cylinder_Rect( points ):
    '''
    rectify cylinders z-axis
    :Args:
        - points: points' coordinates
    :Returns:
        - pnt: rectified points
    '''
    
#     normal calculation
    covCell = np.cov( points.T )  # covariance matrix of pointset
    eigVal, eigVec = np.linalg.eig( covCell )  # eigVal and eigVec of the covariance matrix
    
    maxVal = np.where( eigVal == eigVal.max() )[0][0] 
    minVal = np.where( eigVal == eigVal.min() )[0][0] 
    
    if eigVal[maxVal] / np.sum( eigVal ) >= 0.65: 
        R1 = Rotation_Matrix( eigVec[:, maxVal], np.array( [0, 0, 1] ) )
        minVec = ( R1.dot( np.expand_dims( eigVec[:, minVal], 1 ) ) ).T[0]
        R2 = Rotation_Matrix( minVec, np.array( [1, 0, 0] ) )
        
        R = R2.dot( R1 )
    #     norm_ind = np.where( eigVal == np.min( eigVal ) )
    #     norm = ( eigVec[:, norm_ind] ).T[0][0]  # cylinder normal
    #     
    # #     rectify cylinder's axis
    #     ax = ( np.zeros( ( 1, 3 ) ) )[0]
    #     ax[( np.where( np.abs( norm ) == np.max( np.abs( norm ) ) ) )[0]] = 1
    #     alpha = np.arccos( ( np.sum( norm * ax ) ) / np.linalg.norm( norm ) )
    #     
    #     R1 = Rotation_Matrix( norm, ax )
    #     pnt = np.dot( R1, points.T ).T
    #     
    # #     rotate cylinder's axis to z-axis
    #     axisCyl = ( np.where( eigVal == np.max( eigVal ) ) )[0]
    #     axInd = ( np.where( np.abs( eigVec[:, axisCyl] ) == np.max( np.abs( eigVec[:, axisCyl] ) ) ) )[0]
    #     R2 = np.eye( ( 3 ) )
    #     if axInd != 2:
    #         ax_curr = ( np.zeros( ( 1, 3 ) ) )[0]
    #         ax_curr[axInd] = 1
    #                 
    #         zInd1 = np.where( eigVal != np.max( eigVal ) )
    #         zInd2 = np.where( eigVal[zInd1[0]] > np.min( eigVal[zInd1[0]] ) )
    #         axZ = ( np.zeros( ( 1, 3 ) ) )[0]
    #         axZ[zInd1[0][zInd2[0]]] = 1
    #         
    #         R2 = Rotation_Matrix( ax_curr, axZ )
    #         pnt = np.dot( R2, pnt.T ).T
    #     
    #     return np.dot( R1.T, R2.T ), pnt
        return R, ( R.dot( points.T ) ).T
    else: return None
        


def Fit_Cylinder( x, y, ( a, b, t ), f ):
    '''
    Cylinder fitting
     
    :Args:
        - x: X-axis values
        - y: Y-axis values
        - a,b,t: cylinder parameters
        - f: output file
     
    :Returns:
        - x,y,x0: corrected values and adjusted coefficients
     
    '''
    n = 2 * x.shape[0]
    x0 = np.vstack( ( 0, 0, a, b, t ) )
    xt = x - x0[0]
    yt = y - x0[1]
    ct = np.cos( x0[4] )
    st = np.sin( x0[4] )
    A = np.hstack( ( -2 * ct * ( xt * ct + yt * st ) / x0[2] ** 2 + 2 * st * ( yt * ct - xt * st ) / x0[3] ** 2,
                   - 2 * st * ( xt * ct + yt * st ) / x0[2] ** 2 - 2 * ct * ( yt * ct - xt * st ) / x0[3] ** 2,
                   - 2 * ( xt * ct + yt * st ) ** 2 / x0[2] ** 3, -2 * ( yt * ct - xt * st ) ** 2 / x0[3] ** 3,
                   2 * ( ( xt * ct + yt * st ) * ( -xt * st + yt * ct ) ) / x0[2] ** 2 + 2 * ( ( yt * ct - xt * st ) * ( -xt * ct - yt * st ) ) / x0[3] ** 2 ) )
    B = np.hstack( ( np.diag( ( 2 * ct * ( xt * ct + yt * st ) / x0[2] ** 2 - 2 * st * ( yt * ct - xt * st ) / x0[3] ** 2 )[:, 0] ),
                    np.diag( ( 2 * st * ( xt * ct + yt * st ) / x0[2] ** 2 + 2 * ct * ( yt * ct - xt * st ) / x0[3] ** 2 )[:, 0] ) ) )
     
    flag = 0
    lamb = 1e-3
    V1 = np.zeros( ( x.size, 1 ) )
    it = 0
     
    while flag == 0:
        it += 1
        w = ( xt * ct + yt * st ) ** 2 / x0[2] ** 2 + ( -xt * st + yt * ct ) ** 2 / x0[3] ** 2 - 1
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
     
        if ( np.max( np.abs( dx[0:4] ) ) < 1e-3 and np.max( np.abs( dx[4] ) ) < 1e-5 ):
            sigma2 = np.dot( V.T, V ) / ( n - 5 )
            acc = np.sqrt( np.diag( sigma2 * np.linalg.inv( N ) ) )
            RMSE = np.sqrt( np.dot( V.T, V ) / ( n ) )
            flag = 1
         
        xt = x - x0[0]
        yt = y - x0[1]
        ct = np.cos( x0[4] )
        st = np.sin( x0[4] )  
        A = np.hstack( ( -2 * ct * ( xt * ct + yt * st ) / x0[2] ** 2 + 2 * st * ( yt * ct - xt * st ) / x0[3] ** 2,
                   - 2 * st * ( xt * ct + yt * st ) / x0[2] ** 2 - 2 * ct * ( yt * ct - xt * st ) / x0[3] ** 2,
                   - 2 * ( xt * ct + yt * st ) ** 2 / x0[2] ** 3, -2 * ( yt * ct - xt * st ) ** 2 / x0[3] ** 3,
                   2 * ( ( xt * ct + yt * st ) * ( -xt * st + yt * ct ) ) / x0[2] ** 2 - 2 * ( ( yt * ct - xt * st ) * ( xt * ct + yt * st ) ) / x0[3] ** 2 ) )
        B = np.hstack( ( np.diag( ( 2 * ct * ( xt * ct + yt * st ) / x0[2] ** 2 - 2 * st * ( yt * ct - xt * st ) / x0[3] ** 2 )[:, 0] ),
                        np.diag( ( 2 * st * ( xt * ct + yt * st ) / x0[2] ** 2 + 2 * ct * ( yt * ct - xt * st ) / x0[3] ** 2 )[:, 0] ) ) )
    
    f.write( '(a [m], b [m]): ' + str( x0.T[0][2:4] ) + '\n' )
    f.write( 'Accuracy [m]: ' + str( acc[2:4] ) + '\n' )         
    f.write( 'RMSE [m]: ' + str( RMSE[0][0] ) + '\n' )
    f.write( 'Absolute Error Mean [m]: ' + str( np.mean( np.abs( V ) ) ) + '\n' )
    f.write( 'Maximum Absolute Error [m]: ' + str( np.max( np.abs( V ) ) ) + '\n' )
    
    return x + V[0:V.size / 2], y + V[V.size / 2::], x0
 
def Fit_Cylinder_updated( x, y, r, f1, f ):
    '''
    Cylinder fitting
     
    :Args:
        - x: X-axis values
        - y: Y-axis values
        - a,b,t: cylinder parameters
        - f: output file
     
    :Returns:
        - x,y,x0: corrected values and adjusted coefficients
     
    '''
    r = np.abs( r )
    n = 2 * x.shape[0]
    x0 = np.vstack( ( 0, 0, r ) )
    xt = x - x0[0]
    yt = y - x0[1]

    A = np.hstack( ( -2 * xt, -2 * yt, np.ones( xt.shape ) * ( -2 * x0[2] ) ) )
    B = np.hstack( ( np.diag( ( 2 * xt )[:, 0] ),
                    np.diag( ( 2 * yt )[:, 0] ) ) )
     
    flag = 0
    lamb = 1e-3
    V1 = np.zeros( ( x.size, 1 ) )
    it = 0
     
    while flag == 0:
        it += 1
        w = xt ** 2 + yt ** 2 - x0[2] ** 2
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
     
        if np.max( np.abs( dx[0:3] ) ) < 1e-5:
            if it > 1:
                if np.max( np.max( np.abs( V1 - V ) ) ) < 1e-5 :
                    sigma2 = np.dot( V.T, V ) / ( n - 3 )
                    acc = np.sqrt( np.diag( sigma2 * np.linalg.inv( N ) ) )
                    RMSE = np.sqrt( np.dot( V.T, V ) / ( n ) )
                    flag = 1
            else:
                sigma2 = np.dot( V.T, V ) / ( n - 3 )
                acc = np.sqrt( np.diag( sigma2 * np.linalg.inv( N ) ) )
                RMSE = np.sqrt( np.dot( V.T, V ) / ( n ) )
                flag = 1
         
        xt = x - x0[0]
        yt = y - x0[1] 
        A = np.hstack( ( -2 * xt, -2 * yt, np.ones( xt.shape ) * ( -2 * x0[2] ) ) )
        B = np.hstack( ( np.diag( ( 2 * xt )[:, 0] ),
                    np.diag( ( 2 * yt )[:, 0] ) ) )
    
    d = np.sqrt( V[V.size / 2::] ** 2 + V[0:V.size / 2] ** 2 )
    
    lenV = len( d )
    v_sort = np.sort( np.abs( d ) )
    five_per = np.int32( np.ceil( lenV * 0.05 ) )
    ten_per = np.int32( np.ceil( lenV * 0.10 ) )
    
    f.write( str( np.abs( x0[2, 0] ) ) + '\t' )
#     f.write( str( acc[2] ) + '\t' )         
    f.write( str( np.sqrt( np.dot( d.T, d ) / ( n / 2 ) )[0][0] ) + '\t' )
#     f.write( str( np.mean( np.abs( d ) ) ) + '\t' )
    f.write( str( np.max( np.abs( d ) ) ) + '\t' )
    f.write( str( np.mean( v_sort[lenV - five_per::] ) ) + '\t' )
    f.write( str( np.mean( v_sort[lenV - ten_per::] ) ) + '\n' )
    
    
    f1.write( 'r [m]: ' + str( np.abs( x0[2, 0] ) ) + '\n' )
#     f1.write( 'Accuracy [m]: ' + str( acc[2] ) + '\n' )         
    f1.write( 'RMSE [m]: ' + str( np.sqrt( np.dot( d.T, d ) / ( n / 2 ) )[0][0] ) + '\n' )
#     f1.write( 'Absolute Error Mean [m]: ' + str( np.mean( np.abs( d ) ) ) + '\n' )
    f1.write( 'Maximum Absolute Error [m]: ' + str( np.max( np.abs( d ) ) ) + '\n' )
    f1.write( '5% Absolute Error: ' + str( np.mean( v_sort[lenV - five_per::] ) ) + '\n' )
    f1.write( '10% Absolute Error: ' + str( np.mean( v_sort[lenV - ten_per::] ) ) + '\n' )
       
#     lenV = len( V )
#     v_sort = np.sort( np.abs( V ) )
#     five_per = np.int32( np.ceil( lenV * 0.05 ) )
#     ten_per = np.int32( np.ceil( lenV * 0.10 ) )
#     
#     f.write( str( np.abs( x0[2, 0] ) ) + '\t' )
# #     f.write( str( acc[2] ) + '\t' )         
#     f.write( str( RMSE[0][0] ) + '\t' )
#     f.write( str( np.mean( np.abs( V ) ) ) + '\t' )
#     f.write( str( np.max( np.abs( V ) ) ) + '\t' )
#     f.write( str( np.mean( v_sort[lenV - five_per::] ) ) + '\t' )
#     f.write( str( np.mean( v_sort[lenV - ten_per::] ) ) + '\n' )
#     
#     
#     f1.write( 'r [m]: ' + str( np.abs( x0[2, 0] ) ) + '\n' )
# #     f1.write( 'Accuracy [m]: ' + str( acc[2] ) + '\n' )         
#     f1.write( 'RMSE [m]: ' + str( RMSE[0][0] ) + '\n' )
#     f1.write( 'Absolute Error Mean [m]: ' + str( np.mean( np.abs( V ) ) ) + '\n' )
#     f1.write( 'Maximum Absolute Error [m]: ' + str( np.max( np.abs( V ) ) ) + '\n' )
#     f1.write( '5% Absolute Error: ' + str( np.mean( v_sort[lenV - five_per::] ) ) + '\n' )
#     f1.write( '10% Absolute Error: ' + str( np.mean( v_sort[lenV - ten_per::] ) ) + '\n' )
       
    return x + V[0:V.size / 2], y + V[V.size / 2::], x0 
 
 
def Initial_guess( x, y, z ):
    '''
    Initial guess for the parameters
     
    :Args:
        - x: X-axis values
        - y: Y-axis values
        - z: Z-axis values
     
    :Returns:
        - a,b,t: initial guess for the parameters 
     
    '''
     
#     cylinder slice
    p4_sl = ( np.max( z ) + np.min( z ) ) / 2
    ind1 = np.where( z >= p4_sl - 0.04 )[0]
    ind2 = np.where( z[ind1, :] <= p4_sl + 0.04 )[0]
    x_i = np.repeat( x[ind1[ind2], :], ind1[ind2].size, 1 )
    y_i = np.repeat( y[ind1[ind2], :], ind1[ind2].size, 1 )
    
#     radius 
    distance = np.sqrt( ( x_i - x_i.T ) ** 2 + ( y_i - y_i.T ) ** 2 )
    dmax = np.where( distance == np.max( distance ) )
    r = distance[dmax[0][0], dmax[1][0]] / 2
#     angle relatively to x-axis
    maxVec = np.hstack( ( x[dmax[0][0]], y[dmax[0][0]] ) ) - np.hstack( ( x[dmax[1][0]], y[dmax[1][0]] ) )
    t = np.arccos( ( np.sum( maxVec * np.array( [1, 0] ) ) ) / np.linalg.norm( maxVec ) )
    if maxVec[0] < 0 and maxVec[1] < 0:
        t = np.pi - t
    else:
        if maxVec[0] < 0 and maxVec[1] > 0:
            t -= np.pi / 2
        else:
            if maxVec[0] > 0 and maxVec[1] < 0:
                t = np.pi / 2 - t
             
    return r, r * 0.99, t

def Initial_guess_updated( x, y, z ):
    '''
    Initial guess for the parameters
     
    :Args:
        - x: X-axis values
        - y: Y-axis values
        - z: Z-axis values
     
    :Returns:
        - a,b,t: initial guess for the parameters 
     
    '''
     
# #     cylinder slice
#     p4_sl = ( np.max( z ) + np.min( z ) ) / 2
#     ind1 = np.where( z >= p4_sl - 0.04 )[0]
#     ind2 = np.where( z[ind1, :] <= p4_sl + 0.04 )[0]
#     x_i = np.repeat( x[ind1[ind2], :], ind1[ind2].size, 1 )
#     y_i = np.repeat( y[ind1[ind2], :], ind1[ind2].size, 1 )
#     
# #     radius 
#     distance = np.sqrt( ( x_i - x_i.T ) ** 2 + ( y_i - y_i.T ) ** 2 )
    points = np.hstack( ( x, y, z ) )
    dist = lambda pnt, points: distance.cdist( np.expand_dims( pnt, 0 ), points )
    
    dmax = []
    indices = []
    for i in xrange( len( points ) ):
        dtemp = dist( points[i, :], points )
        dmax.append( np.max( dtemp ) )
        indices.append( np.array( [i, np.where( dtemp == np.max( dtemp ) )[0][0]] ) )
    
    dmax = np.asarray( dmax )
    indices = np.asarray( indices )    
    i_max = np.where( dmax == np.max( dmax ) )[0][0]
    i1 = indices[i_max][0]
    i2 = indices[i_max][1]
#     r = distance[dmax[0][0], dmax[1][0]] / 2
# #     angle relatively to x-axis
#     maxVec = np.hstack( ( x[dmax[0][0]], y[dmax[0][0]] ) ) - np.hstack( ( x[dmax[1][0]], y[dmax[1][0]] ) )
#     t = np.arccos( ( np.sum( maxVec * np.array( [1, 0] ) ) ) / np.linalg.norm( maxVec ) )
#     if maxVec[0] < 0 and maxVec[1] < 0:
#         t = np.pi - t
#     else:
#         if maxVec[0] < 0 and maxVec[1] > 0:
#             t -= np.pi / 2
#         else:
#             if maxVec[0] > 0 and maxVec[1] < 0:
#                 t = np.pi / 2 - t


#     i_Xsorted = np.argsort( x, 0 )
#     p1 = ( x[i_Xsorted[0][0]], y[i_Xsorted[0][0]] )
#     p2 = ( x[i_Xsorted[-1][0]], y[i_Xsorted[-1][0]] )
#     pC = ( x[np.int32( np.floor( i_Xsorted[-1][0] / 2 ) )], y[np.int32( np.floor( i_Xsorted[-1][0] / 2 ) )] )
    if np.abs( i1 - i2 ) > 1:
        p1 = ( x[i1], y[i1] )
        p2 = ( x[i2], y[i2] )
        pC = ( x[np.int32( np.floor( ( i1 + i2 ) / 2 ) )], y[np.int32( np.floor( ( i1 + i2 ) / 2 ) )] )
    else:
        p1 = ( x[min( [i1, i2] )], y[min( [i1, i2] )] )
        pC = ( x[max( [i1, i2] )], y[max( [i1, i2] )] )
        p2 = ( x[-1], y[-1] )
    
    # triangle's sides
    a = np.sqrt( ( p2[0] - pC[0] ) ** 2 + ( p2[1] - pC[1] ) ** 2 )
    b = np.sqrt( ( p1[0] - p2[0] ) ** 2 + ( p1[1] - p2[1] ) ** 2 )
    c = np.sqrt( ( pC[0] - p1[0] ) ** 2 + ( pC[1] - p1[1] ) ** 2 )  
    
    A = 0.5 * ( p1[0] * ( pC[1] - p2[1] ) + pC[0] * ( p2[1] - p1[1] ) + p2[0] * ( p1[1] - pC[1] ) )  # The triangle's area including its sign
    r = 1 / ( 4 * A / ( a * b * c ) )
    return r  # , r * 0.99, t      
 
 
if __name__ == '__main__':
    '''
    f(x,y)=(cos(t)*(x-x0) + sin(t)*(y-y0))^2/a^2 + (-sin(t)*(x-x0) + cos(t)*(y-y0))^2/b^2 - 1
    '''
    
    np.set_printoptions( threshold = 'nan' )  # display all the values in the array
    
#     -------------- Data preparation -------------- 
    print "Loading data..."
    R = None  
#     (id,x,y,z)
    dir = r'D:\Documents\Maatz\Boilers' 
    
#     #==================Axis
#     fin = open( dir + '\\axes.txt' )
#     fileLines = fin.read()
#     fin.close()
#                     
#     # Splitting into list of lines 
#     lines = split( fileLines, '\n' )
#     del fileLines          
#                          
#     # Removing the last line if it is empty
#     while( True ):
#         numLines = len( lines )
#         if lines[numLines - 1] == "":
#             numLines -= 1
#             lines = lines[0: numLines]
#         else:
#             break
#     toFloat = lambda x: np.float32( split( x, '\t' ) )        
#     axes = np.asarray( map( toFloat , lines ) )
#      
#     ii = 0
#     #======================   
     
    os.chdir( dir )
    fileNames = glob.glob( "*.pts" )  # take all files of a specific format
    print fileNames
    
    f = open( dir + '\\results.txt', 'w' )
    for fileName in fileNames:
     
        print "Preparing data..."
        pointSet = []
        IOFactory.ReadPts( fileName, pointSet )
        if len( pointSet ) > 1:
            pointSet = IOFactory.MergePntList( pointSet )
        else:
            pointSet = pointSet[0]
        
        print 'Number of points: ', pointSet.Size
        points = pointSet.ToNumpy()
        
        x1 = np.array( [points[:, 0]] ).T
        y1 = np.array( [points[:, 1]] ).T
        z1 = np.array( [points[:, 2]] ).T
        
        # cylinder center
        x0 = ( np.max( x1 ) + np.min( x1 ) ) / 2
        y0 = ( np.max( y1 ) + np.min( y1 ) ) / 2
        z0 = ( np.max( z1 ) + np.min( z1 ) ) / 2
#         x1 = x1 - x0
#         y1 = y1 - y0
#         z1 = z1 - z0
        
        del points
        
        #     -------------- Cylinder rectification -------------- 
        #     cylinder's z axis should be parallel to z axis
#         
# # #         R = Rotation_Matrix( axes[ii, :], np.array( [0, 0, 1] ) )
# #         R = Rotation_Matrix( np.array( [-0.4030, -0.9151, -0.0106] ), np.array( [0, 0, 1] ) )
# #         pnt = ( R.dot( np.vstack( ( x1.T, y1.T, z1.T ) ) ) ).T
# #         ii = +1
#          
#         res = Cylinder_Rect( np.hstack( ( x1, y1, z1 ) ) )
#         if res == None and R != None:
#             pnt = ( R.dot( np.vstack( ( x1.T, y1.T, z1.T ) ) ) ).T
#         elif res != None: 
#             R, pnt = res[0], res[1]
#         else: 
#             print 'FAILED!!!'
#             continue      
#  
# #         R = np.eye( 3 ) 
# #         pnt = np.hstack( ( x1, y1, z1 ) )
#  
#         x = np.array( [pnt[:, 0]] ).T
#         y = np.array( [pnt[:, 1]] ).T  
#         z = np.array( [pnt[:, 2]] ).T  

        x = x1 - x0
        y = y1 - y0
        z = z1 - z0
        del x1, y1, z1
        
#         # cylinder center
#         x0 = ( np.max( x ) + np.min( x ) ) / 2
#         y0 = ( np.max( y ) + np.min( y ) ) / 2
#         z0 = ( np.max( z ) + np.min( z ) ) / 2
#         x = x - x0
#         y = y - y0
#         z = z - z0
        
        # draw data points
#         pointSet_old = PointSet( np.hstack( ( x1, y1, z1 ) ) )
#         fig = Visualization.RenderPointSet( pointSet_old, renderFlag = 'color', color = ( 0, 1, 0 ), pointSize = 2 )
        rectif_pnt = PointSet( np.hstack( ( x, y, z ) ) )
        fig = VisualizationVTK.RenderPointSet(rectif_pnt, renderFlag='color', color=(1, 0, 0), pointSize=2)
#         Visualization.Show()
        
        print "Approximate values calculation..."
#         coeff0 = Initial_guess( x, y, z )  # Initial guess for the parameters
        coeff0 = Initial_guess_updated( x, y, z )
        #     coeff0 = (0.025, 0.026, 0)
        
        #     -------------- Cylinder fitting --------------    
        print "Parameters' calculation..."
        #     least square with Levenberg-Marquardt algorithm
        f1 = open( dir + '\\' + fileName[0:-4] + '_output.txt', 'w' )
        f.write( fileName[0:-4] + '\t' )
#         f = open(fileName[0:-4] + '_output.txt', 'w')
        x_ney, y_new, coeff = Fit_Cylinder_updated( x, y, coeff0, f1, f )
    #         x_ney, y_new, coeff = Fit_Cylinder(x, y, coeff0, f)
        #     draw fitted points
        pointSet_new = PointSet( np.hstack( ( x_ney, y_new, z ) ) )
        fig = VisualizationVTK.RenderPointSet(pointSet_new, renderFlag='color', _figure=fig, color=(0, 0, 1),
                                              pointSize=2)
        
#         #     -------------- Normals -------------- 
#         pnt = np.hstack( ( x, y, z ) )
# #         norm = pnt - np.hstack( ( np.repeat( coeff[0:2].T, pnt.shape[0], 0 ), np.expand_dims( pnt[:, 2], 1 ) ) )
#         n = pnt - np.hstack( ( np.repeat( coeff[0:2].T, pnt.shape[0], 0 ), np.expand_dims( pnt[:, 2], 1 ) ) )
#         norm = np.dot( R, n.T ).T
#         normals = NormalsProperty( pointSet, norm )
#         Visualization.RenderPointSet( normals, renderFlag = 'color', pointSize = 3 )
#          
#         #     -------------- Incidence angle -------------- 
#         incidence_angles = np.rad2deg( np.arccos( np.sum( pointSet.ToNumpy() * norm, 1 ) / ( np.linalg.norm( pointSet.ToNumpy(), 2, 1 ) * np.linalg.norm( norm, 2, 1 ) ) ) ) 
#         f1.write( 'Incidence Angles [deg]: ' + '\n' + str( incidence_angles ) )
        
        f1.close()
#         Visualization.Show()
    
    f.close()
    VisualizationVTK.Show()
