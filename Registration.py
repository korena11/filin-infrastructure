'''
Created on 10 march 2014

@author: vera
'''

import numpy as np
import vtk as vtk
from IOFactory import IOFactory
from PointSet import PointSet
from Visualization import Visualization 


class Registration:

    @classmethod
    def Rotation_Quaternions( cls, axis, theta ):
        '''
        create rotation matrix given axis and angle
        
        :Args:
            - axis: axis to rotate around
            - theta: rotation angle in degrees
            
        :Returns:
            - R: rotation matrix
        
        '''
        s = np.sin( np.radians( theta ) )
        c = np.cos( np.radians( theta ) )
        t = 1 - c
        x = axis[0]
        y = axis[1]
        z = axis[2]
        R = np.array( [[t * x ** 2 + c, t * x * y + s * z, t * x * z - s * y],
                  [t * x * y - s * z, t * y ** 2 + c, t * y * z + s * x],
                  [t * x * z + s * y, t * y * z - s * x, t * z ** 2 + c]] )
        return R
    
    
    @classmethod
    def PointSet2Array( cls, fileName ):
        '''
        create source and target pointLists 
        
        :Args: 
            - fileName: points file name
            
        :Returns: 
            - Pnt: points as nx3 ndarray
        
        '''
        pointSetList = []
        vert = []
        if fileName[-3:] == 'pts':
            n = IOFactory.ReadPts( fileName, pointSetList, vertList = vert )
        else:
            n = IOFactory.ReadXYZ( fileName, pointSetList )
        
        Pnt = pointSetList[0].ToNumpy()
        if n != 1:
            for i in np.arange( 1, n + 1 ):
                tempP = Pnt
                data = pointSetList[i].ToNumpy()
                Pnt = np.vstack( ( tempP, data ) )
                        
#        pointSet_Res = PointSet(sPnt) 
#        fig1 = Visualization.RenderPointSet(pointSet_Res, renderFlag='color', color=(1, 0, 0))
#        pointSet_Target = PointSet(targetPnt)
#        fig1 = Visualization.RenderPointSet(pointSet_Target, renderFlag='color', _figure=fig1, color=(0, 0, 1))
#        Visualization.Show()

        return Pnt
       
       
    @classmethod
    def TransformData( cls, points, translation, rot_angle, rot_axis ): 
        '''
        transform points according to given translation and rotation
        
        :Args:
            - points: pointset to transform
            - translation: translation vector 3x1
            - rot_angle: rotation angle in degrees
            - rot_axis: rotation axis
            
        :Returns:
            - transformedPnt: points after transformation
        
        '''
    
        # if vertices1.size>0 and vertices1.size>0:
        
        rotation = Registration.Rotation_Quaternions( rot_axis, rot_angle )  # (axis, angle)
#         transformedPnt = translation.T + np.dot( points, rotation )
        transformedPnt = ( rotation.T.dot( points.T ) + translation ).T
        
#        pointSet_Res = PointSet(sourcePnt) 
#        fig2 = Visualization.RenderPointSet(pointSet_Res, renderFlag='color', color=(0, 1, 0))
#        fig2 = Visualization.RenderPointSet(pointSet_Target, renderFlag='color', _figure=fig2, color=(0, 0, 1))
#        Visualization.Show()  
        return transformedPnt
        
    @classmethod
    def Registration2D( cls, xy_world, xy_local ):
        '''
        2D transformation
        
        :Args: 
            - xy_world: pointset in coordinate system we want to transform to
            - xy_local: pointset in coordinate system we want to transform from
        
        :Returns: transformed points file
        
        '''
        tlsSize = xy_world.size
    
        # ============ Build Matrices ==============
        a1 = np.kron( np.ones( ( tlsSize / 2, 1 ) ), np.array( [[1], [0]] ) )
        a2 = np.kron( np.ones( ( tlsSize / 2, 1 ) ), np.array( [[0], [1]] ) )
        a3 = np.reshape( xy_local, ( tlsSize, 1 ) )
        reordered_tls = np.hstack( ( np.array( [xy_local[:, 1]] ).T, np.array( [-xy_local[:, 0]] ).T ) );
        a4 = np.reshape( reordered_tls, ( tlsSize, 1 ) )
        A = np.hstack( ( a1, a2, a3, a4 ) )
        # print A
        
        L = np.reshape( xy_world, ( tlsSize, 1 ) )
        
        # print L
        
        # ============ Adjustment ==============
        
        N = np.dot( A.T , A )
        Ninv = np.linalg.inv( N )
        u = np.dot( A.T , L ) 
        x = np.dot( Ninv, u )  # (tx, ty, cos(a), sin(a))
        t = np.array( x[0:2] )
        R = np.array( [[x[2, 0], x[3, 0]], [-x[3, 0], x[2, 0]]] )
        # print x
        
        # ============ Data to transform (x, y) ==============
        data = np.loadtxt( r'D:\My Documents\saflulim\Minna Evron\MinaEvron_data\points_VIII.txt', delimiter = '\t', skiprows = 1 )
        xyLocal = data[0:, 1:3]
     
        # ============ Planar Transformation ==============
        xyWorld = ( t + np.dot( R, xyLocal.T ) ).T 
        xyzWorld = np.hstack( ( np.array( [np.arange( 0, xyWorld.shape[0] )] ).T, xyWorld, np.array( [data[0:, 3]] ).T - 4.89 ) )  # x, y transformed and original z
         
#         # y rotation
#        x = np.array([xyzWorld[:, 1]]) - np.average(xyzWorld[:, 1]).T
#        z = np.array([xyzWorld[:, 3]]) - np.average(xyzWorld[:, 3]).T
#        x_N = (z * np.sin(np.radians(5)) + x * np.cos(np.radians(5))).T
#        z_N = (z * np.cos(np.radians(5)) - x * np.sin(np.radians(5))).T
#        
#        xyzWorld = np.hstack((np.array([xyzWorld[:, 0]]).T, x_N + np.average(xyzWorld[:, 1]), np.array([xyzWorld[:, 2]]).T, z_N + np.average(xyzWorld[:, 3])))
#
#         # x rotation        
#        y = np.array([xyzWorld[:, 2]]) - np.average(xyzWorld[:, 2]).T
#        z = np.array([xyzWorld[:, 3]]) - np.average(xyzWorld[:, 3]).T
#        y_N = (y * np.cos(np.radians(-5)) - z * np.sin(np.radians(-5))).T
#        z_N = (y * np.sin(np.radians(-5)) + z * np.cos(np.radians(-5))).T
#        
#        xyzWorld = np.hstack((xyzWorld[:, 0:2], y_N + np.average(xyzWorld[:, 2]), z_N + np.average(xyzWorld[:, 3])))
         
        # ============ Write the transformed coordinates to text file ==============
        np.savetxt( r'D:\My Documents\saflulim\Minna Evron\MinaEvron_data\after2Dreg\VIII_World.txt', xyzWorld, fmt = ['%d', '%.4f', '%.4f', '%.4f'], delimiter = '\t', newline = '\n' )
        print 'done'
         
#        L = np.reshape(xyzWorld[:, 1:], (xyzWorld.shape[0] * 3, 1))  # [x,y,z,x,y,z...]'
#        A = np.zeros((xyzWorld.shape[0] * 3, 12))
#        A[::3, 0:3] = data[0:, 1:4]
#        A[1::3, 3:6] = data[0:, 1:4]
#        A[2::3, 6:9] = data[0:, 1:4]
#        A[:, 9::] = np.kron(np.ones((xyzWorld.shape[0], 1)), np.eye((3)))
#        x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, L))
#        return x
        
    
        
    @classmethod
    def ICP( cls, origPnt, targetPnt ):  
        '''
        algorithm ICP
        
        :Args: 
            - origPnt: source pointSet
            - sourcePnt: source pointSet subset that is a subset of a target pointset 
            - targetPnt: target pointSet
        
        :Returns: pointSet_Res: source points after ICP transformation 
        
        '''
        # ============ create source points ==============
        print "Creating source points..."

        origPoints = vtk.vtkPoints()
        origVertices = vtk.vtkCellArray()
        
        funcS = lambda v: ( origVertices.InsertNextCell( 1 ), origVertices.InsertCellPoint( origPoints.InsertNextPoint( v ) ) )
        
        map( funcS, origPnt )
       
        origSource = vtk.vtkPolyData()
        origSource.SetPoints( origPoints )
        origSource.SetVerts( origVertices )
        origSource.Update()

        
        # ============ create target points ==============
        print "Creating target points..."
        
        targetPoints = vtk.vtkPoints()
        targetVertices = vtk.vtkCellArray()
        
        funcT = lambda v: ( targetVertices.InsertNextCell( 1 ), targetVertices.InsertCellPoint( targetPoints.InsertNextPoint( v ) ) )
        
        map( funcT, targetPnt )
        
        target = vtk.vtkPolyData()
        target.SetPoints( targetPoints )
        target.SetVerts( targetVertices )
        target.Update()


        print "Running ICP..."
        # ============ run ICP ==============
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource( origSource )
        icp.SetTarget( target )
        icp.GetLandmarkTransform().SetModeToRigidBody()
        # icp.DebugOn()
        icp.SetMaximumNumberOfIterations( 100 )
        icp.StartByMatchingCentroidsOn()
        icp.Modified()
        icp.Update()
        print 'Number of ICP iterations: ', icp.GetNumberOfIterations()
        print 'Mean dist: ', icp.GetMeanDistance()
    
        icpTransformFilter = vtk.vtkTransformPolyDataFilter()
        icpTransformFilter.SetInput( origSource )
     
        icpTransformFilter.SetTransform( icp )
        icpTransformFilter.Update()
        
        
        transformedSource = icpTransformFilter.GetOutput()  
        
        transMat = icp.GetLandmarkTransform().GetMatrix()   
        
        # ============ create results file ==============
        print "Rendering..."
        points = np.zeros( ( origPnt.shape[0], 3 ) )
        map( transformedSource.GetPoint, xrange( origPnt.shape[0] ), points )
        
        points = np.array( points )
        
        pointSet_Res = PointSet( points ) 
        fig = Visualization.RenderPointSet( pointSet_Res, renderFlag = 'color', color = ( 1, 0, 0 ), pointSize = 1 )
        pointSet_Target = PointSet( targetPnt )
        fig = Visualization.RenderPointSet( pointSet_Target, renderFlag = 'color', _figure = fig, color = ( 0, 0, 1 ), pointSize = 1 )
        
#        print "Writing to the file..."
        #=======================================================================
        # f = open('D:\\Desktop\\out.txt', 'w')
        # func = lambda v: (f.write(v[0] + '\t' + v[1] + '\t' + v[2] + '\n'))
        # points = np.char.mod('%08f', points)
        # map(func, points)
        # 
        # f.close()
        #=======================================================================
        
        return transMat


if __name__ == '__main__':
#     #===========================================================================
#     # 2D REGISTRATION
#     #===========================================================================
#     # ============ load data ==============
#     data_registration = np.loadtxt(r'D:\Documents\saflulim\Minna Evron\arcgis\reg\VIII.txt', dtype=np.float32, delimiter='\t')
#     Registration.Registration2D(data_registration[:, 2:], data_registration[:, 0:2])
    
    
    #===========================================================================
    # ICP    
    #===========================================================================
    # ============ load source & target data ==============
    print "Loading data..."     
    fileName_s = 'D:\\Documents\\Pointsets\\rabin_bor2_reg.pts'
    fileName_t = 'D:\\Documents\\Pointsets\\rabin_bor1_reg.pts'  # target pointset 
     
    # approximate registration values 
    transl_4 = np.array( [[2.775], [-20.115], [-2.454]] )
    axis_4 = np.array( [0, 0, 1] )
    angle_4 = -168.442
#     
#     transl_5 = np.array([[1.698], [-32.430], [-2.536]])
#     axis_5 = np.array([0, 0, -1])
#     angle_5 = 62.350  # .578
      
#     transl_4 = np.array([[40.095], [125.339], [-3.313]])
#     axis_4 = np.array([0, 0, 1])
#     angle_4 = 169.490
#     
#     transl_5 = np.array([[0], [0], [0]])
#     axis_5 = np.array([0, 0, 1])
#     angle_5 = 0
      
    print "Source data..." 
#     sData = Registration.PointSet2Array(fileName_s)   
#     sourceData = Registration.TransformData(sData, transl_4, angle_4, axis_4)
    pointSetList = []
    pointSet = IOFactory.ReadPts( fileName_s, pointSetList, merge=True )

    pp = pointSet.ToNumpy()
    # remove duplicate points
    b = np.ascontiguousarray( pp ).view( np.dtype( ( np.void, pp.dtype.itemsize * pp.shape[1] ) ) )
#         _, idx = np.unique( b, return_index = True )
#         pp = pp[idx]      
    sourceData = np.unique( b ).view( pp.dtype ).reshape( -1, pp.shape[1] )  # speed-up the procedure
    del pointSet

    pointSet_s = PointSet( pp )
    sourceData1 = pointSet_s.ToNumpy()
#     sourceData = Registration.PointSet2Array(fileName_s) 
 
    print "Target data..." 
#     tData = Registration.PointSet2Array(fileName_t)
#     targetData = Registration.TransformData(tData, transl_5, angle_5, axis_5)
    pointSetList = []
    pointSet = IOFactory.ReadPts( fileName_t, pointSetList, merge=True )
    pp = pointSet.ToNumpy()
    # remove duplicate points
    b = np.ascontiguousarray( pp ).view( np.dtype( ( np.void, pp.dtype.itemsize * pp.shape[1] ) ) )
#         _, idx = np.unique( b, return_index = True )
#         pp = pp[idx]      
    targetData = np.unique( b ).view( pp.dtype ).reshape( -1, pp.shape[1] )  # speed-up the procedure
    del pointSet
    pointSet_t = PointSet( pp )


    targetData1 = pointSet_t.ToNumpy()
    
    fig_ = Visualization.RenderPointSet( pointSet_t, renderFlag = 'color', color = ( 1, 0, 0 ), pointSize = 1.0 )
    Visualization.RenderPointSet( pointSet_s, renderFlag = 'color', _figure = fig_, color = ( 0, 0, 1 ), pointSize = 1.0 )
    Visualization.Show()
    
    targetData = Registration.TransformData( targetData1, np.array( [[0], [0], [0]] ), 0, [0, 0, 1] )
    sourceData = Registration.TransformData( sourceData1, np.array( [[45], [-10], [0]] ), 222.578, [0, 0, 1] )

#     targetData = Registration.PointSet2Array(fileName_t)
    
     
    pointSet_Res = PointSet( sourceData ) 
    fig1 = Visualization.RenderPointSet( pointSet_Res, renderFlag = 'color', color = ( 0, 0, 1 ), pointSize = 1.0 )
    pointSet_Target = PointSet( targetData )
    Visualization.RenderPointSet( pointSet_Target, renderFlag = 'color', _figure = fig1, color = ( 1, 0, 0 ), pointSize = 1.0 )
    
    Visualization.Show()
    
    print "ICP..." 
    transMat1 = Registration.ICP( sourceData, targetData )
    print transMat1
    mat = map( transMat1.GetElement, np.reshape( np.repeat( np.expand_dims( np.arange( 4 ), 0 ), 4, 0 ), ( 1, 16 ) )[0],
              np.reshape( np.repeat( np.expand_dims( np.arange( 4 ), 1 ), 4, 1 ), ( 1, 16 ) )[0] )
    mat = np.reshape( np.asarray( mat ), ( 4, 4 ) )
     
#     # ====== Big Dataset
#     print "Big Dataset..."  
#     fileName_b = 'D:\\Documents\\Pointsets\\avdat1_14_big.pts'
#     bigData = ( Registration.PointSet2Array( fileName_b ) )  # [0::3, :] 
#     bigData_xyz = np.hstack( ( bigData, np.ones( ( bigData.shape[0], 1 ) ) ) )
#     print "Transforming..."
#     mat = np.array([[0.999828577042, -0.000240735593, -0.018514433876, -0.312845230103],
#                   [0.000041077041, 0.999941825867, -0.010783565231, 0.450944900513],
#                   [0.018515951931, 0.010780955665, 0.999770462513, 0.015809178352],
#                   [0, 0, 0, 1]])
#     bigData_transformed = bigData_xyz.dot(mat)
      
#     print "Writing to the file..."
#     f = open('D:\\Desktop\\avdat1_2014_transformed.txt', 'w')
#     func = lambda v: (f.write(v[0] + ' ' + v[1] + ' ' + v[2] + '\n'))
#     bigData_final = np.char.mod('%08f', bigData_transformed[:, 0:3])
#     f.write(np.str(bigData_transformed.shape[0]) + '\n')
#     map(func, bigData_final)
#        
#     f.close()
    print "Done!"
    
    
#     # ===== second set
#     print "Loading data..."     
#     fileName_s = 'D:\\Documents\\Pointsets\\bonim_4_1.pts'
#     fileName_t = 'D:\\Documents\\Pointsets\\bonim_5_1.pts'  # target pointset 
#      
#     print "Source data..." 
#     sData = Registration.PointSet2Array(fileName_s)   
#     sourceData = Registration.TransformData(sData, transl_4, angle_4, axis_4)
# 
#     print "Target data..." 
#     tData = Registration.PointSet2Array(fileName_t)
#     targetData = Registration.TransformData(tData, transl_5, angle_5, axis_5)
#     
#     pointSet_Res = PointSet(sourceData) 
#     fig4 = Visualization.RenderPointSet(pointSet_Res, renderFlag='color', color=(1, 0, 0), pointSize=2.0)
#     pointSet_Target = PointSet(targetData)
#     Visualization.RenderPointSet(pointSet_Target, renderFlag='color', _figure=fig4, color=(0, 0, 1), pointSize=2.0)
#    
#     print "ICP..." 
#     transMat2 = Registration.ICP(sourceData, targetData)
#     print transMat2
#     
#     fileS = 'D:\\Documents\\Pointsets\\bonim_4_big.pts'
#     fileT = 'D:\\Documents\\Pointsets\\bonim_5_big.pts'
#     print "Big data..." 
#     sData = Registration.PointSet2Array(fileS)   
#     sData = Registration.TransformData(sData, transl_4, angle_4, axis_4)
#  
#     tData = Registration.PointSet2Array(fileT)
#     tData = Registration.TransformData(tData, transl_5, angle_5, axis_5)   
#     
#     sdata = np.hstack((sData, np.ones((sData.shape[0], 1))))
#     pnts = np.dot(sdata, mat)[:, 0:3]
#  
#     pointSet_Res = PointSet(sData) 
#     fig2 = Visualization.RenderPointSet(pointSet_Res, renderFlag='color', color=(1, 1, 0), pointSize=2.0)
#     pointSet_Target = PointSet(tData)
#     Visualization.RenderPointSet(pointSet_Target, renderFlag='color', _figure=fig2, color=(0, 1, 0), pointSize=2.0)
#      
#     pointSet_Res = PointSet(pnts)
#     fig3 = Visualization.RenderPointSet(pointSet_Res, renderFlag='color', color=(1, 1, 0), pointSize=2.0)
#     Visualization.RenderPointSet(pointSet_Target, renderFlag='color', _figure=fig3, color=(0, 0, 1), pointSize=2.0)
    
    Visualization.Show()
