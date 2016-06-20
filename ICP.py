import vtk as vtk
# from vtk import *
import numpy as np
from IOFactory import IOFactory

def Rotation_Quaternions(axis, theta):
    '''
    create rotation matrix given axis and angle
    '''
    s = np.sin(np.radians(theta))
    c = np.cos(np.radians(theta))
    t = 1 - c
    x = axis[0]
    y = axis[1]
    z = axis[2]
    R = np.array([[t * x ** 2 + c, t * x * y + s * z, t * x * z - s * y],
                  [t * x * y - s * z, t * y ** 2 + c, t * y * z + s * x],
                  [t * x * z + s * y, t * y * z - s * x, t * z ** 2 + c]])
    return R



if __name__ == '__main__':
    
    print "Loading data..."
# ============ load source data ==============
    fileName = 'Source_data.pts'
    pointSetList1 = []
    vert1 = []
    nS = IOFactory.ReadPts(fileName, pointSetList1, vert1)
    vertices1 = np.asarray(vert1)
    dataSource = np.asarray(pointSetList1)
    sPnt = dataSource[:, 0:2]
# ============ load target data ==============
    fileName = 'Target_data.pts'
    pointSetList2 = []
    vert2 = []
    nT = IOFactory.ReadPts(fileName, pointSetList2, vert2)
    vertices2 = np.asarray(vert2)
    dataTarget = np.asarray(pointSetList1)
    targetPnt = dataTarget[:, 0:2]

   
#============ approximate rotation and translation ==============
    #===========================================================================
    # sPnt = np.array([[1.0, 1.0, 0], [0.0, 1.1, 1.2], [0, 0.8, 1.0]])
    # targetPnt = np.array([[1.0, 1.1, 0], [0.2, 1.1, 1.2], [0, 0.9, 1.1]])
    #===========================================================================
    
    #--------------------------------- if vertices1.size>0 and vertices1.size>0:
        
    Translation = np.array([[238784.649], [583647.344], [-403.113]])  # (xt, yt, zt)
    Rotation = Rotation_Quaternions(np.array([0, 0, 1]), 89.834)  # (axis, angle)
    sourcePnt = (Translation + np.dot(Rotation.T, sPnt.T)).T
    

 
# ============ create source points ==============
    print "Creating source points..."

    sourcePoints = vtk.vtkPoints()
    sourceVertices = vtk.vtkCellArray()

    funcS = lambda v: (sourceVertices.InsertNextCell(1), sourceVertices.InsertCellPoint(sourcePoints.InsertNextPoint(v)))
    
    map(funcS, sourcePnt)
       
    source = vtk.vtkPolyData()
    source.SetPoints(sourcePoints)
    source.SetVerts(sourceVertices)
    source.Update()
     
#===============================================================================
#    print "Displaying source points..."
# # ============ display source points ==============
#    pointCount = 3
#    for index in range(pointCount):
#        point = [0, 0, 0]
#        sourcePoints.GetPoint(index, point)
#        print "source point[%s]=%s" % (index, point)
#===============================================================================
    
# ============ create target points ==============
    print "Creating target points..."
    
    targetPoints = vtk.vtkPoints()
    targetVertices = vtk.vtkCellArray()

    funcT = lambda v: (targetVertices.InsertNextCell(1), targetVertices.InsertCellPoint(targetPoints.InsertNextPoint(v)))
    
    map(funcT, targetPnt)
    
    target = vtk.vtkPolyData()
    target.SetPoints(targetPoints)
    target.SetVerts(targetVertices)
    target.Update()
 
 
 #==============================================================================
 #   # ============ display target points ==============
 #   print "Displaying target points..."
 #   pointCount = 3
 #   for index in range(pointCount):
 #       point = [0, 0, 0]
 #       targetPoints.GetPoint(index, point)
 #       print "target point[%s]=%s" % (index, point)
 # 
 #==============================================================================


    print "Running ICP ----------------"
# ============ run ICP ==============
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    # icp.DebugOn()
    icp.SetMaximumNumberOfIterations(20)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    icpTransformFilter = vtk.vtkTransformPolyDataFilter()
    icpTransformFilter.SetInput(source)
 
    icpTransformFilter.SetTransform(icp)
    icpTransformFilter.Update()
 
    transformedSource = icpTransformFilter.GetOutput()
    
     
    points = []
    point = [0, 0, 0]
    func = lambda v: (transformedSource.GetPoint(v, point), points.append(np.array(point)))
    map(func, range(3))
    
    points = np.array(points)
    
    f = open('D:\\Desktop\\out.txt', 'w')
    func = lambda v: (f.write(v[0] + '\t' + v[1] + '\t' + v[2] + '\n'))
    points = np.char.mod('%08f', points)
    map(func, points)

    f.close()
 
# ============ display transformed points ==============
    pointCount = 3
    for index in range(pointCount):
            point = [0, 0, 0]
            transformedSource.GetPoint(index, point)
            print "transformed source point[%s]=%s" % (index, point)
