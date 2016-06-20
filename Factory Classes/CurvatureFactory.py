# from CurvatureProperty import CurvatureProperty
from NeighborsFactory import NeighborsFactory
from RotationMatrixFactory import RotationMatrixFactory  
import numpy as np


class CurvatureFactory( object ):
    '''
    curvature parameters computation
    '''
    
    @staticmethod
    def __GoodPoint( points, rad ):
        '''
        determine weather the point is appropriate for curvature calculation
        '''

        sector = 0
        pAngle = np.zeros( ( 1, points.shape[0] ) )
        ind1 = np.where( np.abs( points[:, 0] ) > 1e-6 )[0]
        pAngle[0, ind1] = np.arctan2( points[ind1, 1], points[ind1, 0] )
        ind2 = np.where( np.abs( points[:, 0] ) <= 1e-6 )[0]
        if ind2.size != 0:
            ind2_1 = np.where( points[ind2, 1] > 0 )[0]
            ind2_2 = np.where( points[ind2, 1] < 0 )[0]
            if ind2_1.size != 0:
                pAngle[0, ind2[ind2_1]] = np.pi / 2.0
            if ind2_2.size != 0:
                pAngle[0, ind2[ind2_2]] = 3.0 * np.pi / 2.0
        
        pAngle[np.where( pAngle < 0 )] += 2 * np.pi
        for i in np.linspace( 0, 7.0 * np.pi / 4.0, 8 ):
            pInSector = ( np.where( pAngle[np.where( pAngle <= i + np.pi / 4.0 )] > i ) )[0].size
            if pInSector >= rad * 85:
                sector += 1
        
    #     print sector        
        if sector >= 7:
            return 1
        else:
            return 0
    
    @staticmethod
    def __BiQuadratic_Surface( points ):
        '''
        BiQuadratic surface adjustment to discrete point cloud
        :Args: 
            - points: (nx3 array) 3D points coordinates
        :Returns:
            - p: (array) surface's coefficients
        '''
        # ==== initial guess ====
        x = np.expand_dims( points[:, 0], 1 )
        y = np.expand_dims( points[:, 1], 1 )
        z = np.expand_dims( points[:, 2], 1 )
        
        A = np.hstack( ( x ** 2, y ** 2, x * y, x, y, np.ones( x.shape ) ) )
        p = np.dot( np.linalg.inv( np.dot( A.T, A ) ), np.dot( A.T, z ) )
        
        return p

    @staticmethod
    def __PCA( points, rad = None ):
        '''
        compute eigenvalues and eigenvectors
        :Args: 
            - points: ndarray nx3 points with origin in pnt
        '''    
        
        if rad != None:
            sigma = rad / 3
            w = ( 1 / np.sqrt( 2 * np.pi * sigma ** 2 ) ) * np.exp( -( points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2 ) / ( 2 * sigma ** 2 ) )
            if np.sum( np.isnan( w ) ) > 0 or np.sum( np.isinf( w ) ) > 0 or np.abs( np.sum( w ) ) < 1e-10:
                w = np.ones( points[:, 0].shape )
        else: w = np.ones( points[:, 0].shape )
        
        pT = np.vstack( ( np.expand_dims( w * points[:, 0], 0 ), np.expand_dims( w * points[:, 1], 0 ), np.expand_dims( w * points[:, 2], 0 ) ) )
        
        C = np.dot( pT, points ) / np.sum( w )  # points.shape[0]  # covariance matrix of pointset
        eigVal, eigVec = np.linalg.eig( C )
        
        return eigVal, eigVec
    
    @staticmethod
    def Curvature_FundamentalForm( pnt, points, rad, tree ):
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
        neighbor = NeighborsFactory.GetNeighborsIn3dRange_KDtree( pnt, points, rad, tree )
        neighbors = neighbor.ToNumpy()
        if neighbors[1::, :].shape[0] > 5:
            neighbors = ( neighbors - np.repeat( np.expand_dims( pnt, 0 ), neighbors.shape[0], 0 ) )[1::, :]
            eigVal, eigVec = CurvatureFactory.__PCA( neighbors, rad )
            
            normP = eigVec[:, np.where( eigVal == np.min( eigVal ) )[0][0]]
            # if a normal of a neighborhood is in an opposite direction rotate it 180 degrees
            if np.linalg.norm( pnt, 2 ) < np.linalg.norm( pnt + normP, 2 ):
                normP = -normP
            n = np.array( [0, 0, 1] )
            
            # rotate the neighborhood to xy plane 
            rot_mat = RotationMatrixFactory.Rotation_2Vectors( normP, n )
            
            neighbors = ( np.dot( rot_mat, neighbors.T ) ).T
            pnt = np.array( [0, 0, 0] )
            pQuality = CurvatureFactory.__GoodPoint( neighbors, rad )
            
        #     fig1 = Visualization.RenderPointSet(PointSet(points), renderFlag='height', pointSize=4)
        #     Visualization.RenderPointSet(PointSet(np.expand_dims(pnt, 0)), renderFlag='color', _figure=fig1, color=(0, 0, 0), pointSize=6)
        #     Visualization.Show()
            if  pQuality == 1:
                p = CurvatureFactory.__BiQuadratic_Surface( np.vstack( ( neighbors, pnt ) ) )
                Zxx = 2 * p[0]
                Zyy = 2 * p[1]
                Zxy = p[2]
             
                k1 = ( ( ( Zxx + Zyy ) + np.sqrt( ( Zxx - Zyy ) ** 2 + 4 * Zxy ** 2 ) ) / 2 )[0]
                k2 = ( ( ( Zxx + Zyy ) - np.sqrt( ( Zxx - Zyy ) ** 2 + 4 * Zxy ** 2 ) ) / 2 )[0]  
            else:
                k1, k2 = -999, -999
        else:
            k1, k2 = -999, -999
        
        return np.array( [k1, k2] )
    
    @staticmethod
    def Similarity_Curvature( k1, k2 ):
        '''
        calculates similarity curvature (E,H)
        :Args:
            - k1,k2: principal curvatures (k1>k2)
        :Returns:
            - similarCurv: values of similarity curvature (E,H)
            - rgb: RGB color for every point
        '''
#         if 'points' in kwargs and ('k1' not in kwargs and 'k2' not in kwargs):
#             points = kwargs['points']
#             curv = np.asarray( map( functools.partial( CurvatureFactory.Curvature_FundamentalForm, points = pointSet, rad = coeff, tree = tree ), pp ) )
        
        k3 = np.min( ( np.abs( k1 ), np.abs( k2 ) ), 0 ) / np.max( ( np.abs( k1 ), np.abs( k2 ) ), 0 )
        similarCurv = np.zeros( ( k3.shape[0], 2 ) )
        rgb = np.zeros( ( k3.shape[0], 3 ), dtype = np.float32 )
        
        sign_k1 = np.sign( k1 )
        sign_k2 = np.sign( k2 )
        signK = sign_k1 + sign_k2
        
        # (+,0) 
        positive = np.where( signK == 2 )
        similarCurv[positive[0], 0] = k3[positive]
        rgb[positive[0], 0] = k3[positive]
        # (-,0)
        negative = np.where( signK == -2 )
        similarCurv[negative[0], 0] = -k3[negative]
        rgb[negative[0], 1] = k3[negative]
        
        dif = ( np.where( signK == 0 ) )[0]
        valueK = np.abs( k1[dif] ) >= np.abs( k2[dif] )
        k2_k1 = np.where( valueK == 1 )
        k1_k2 = np.where( valueK == 0 )
        # (0,+)
        similarCurv[dif[k2_k1[0]], 1] = ( k3[dif[k2_k1[0]]].T )[0]
        rgb[dif[k2_k1[0]], 0:2] = np.hstack( ( k3[dif[k2_k1[0]]], k3[dif[k2_k1[0]]] ) )
        # (0,-)
        similarCurv[dif[k1_k2[0]], 1] = -( k3[dif[k1_k2[0]]].T )[0]
        rgb[dif[k1_k2[0]], 2] = ( k3[dif[k1_k2[0]]].T )[0]
        
        return rgb, similarCurv
    
    @staticmethod
    def Mean_Curvature( k1, k2 ):
        '''
        '''
        return ( k1 + k2 ) / 2
        
        
    @staticmethod
    def Gaussian_Curvature( k1, k2 ):
        '''
        '''
        return k1 * k2 
        
    @staticmethod
    def Curvadness( k1, k2 ):
        '''
        '''
        return np.sqrt( ( k1 ** 2 + k2 ** 2 ) / 2 )

    @staticmethod
    def Shape_Index( k1, k2 ):
        '''
        '''
        shapeI = np.zeros( k1.shape )
        equalZero = np.where( np.abs( k1 - k2 ) <= 1e-6 )[0]
        difZero = np.where( k1 != k2 )[0]
        if equalZero.size != 0:
            shapeI[equalZero, :] = 0
        shapeI[difZero, :] = ( 1.0 / np.pi ) * np.arctan2( ( k2 + k1 )[difZero], ( k2 - k1 )[difZero] )  
        return shapeI
    
