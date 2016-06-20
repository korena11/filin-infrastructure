'''
Created on 17 June 2014

@author: Vera
'''

import numpy as np
import numpy.matlib as matlib
from Registration import Registration
from PointSet import PointSet
from Visualization import Visualization 
from NormalsFactory import NormalsFactory
from PointSet import PointSet
from IOFactory import IOFactory
from mayavi import mlab
from matplotlib.path import Path
from PointSubSet import PointSubSet
import PCA_gm
from TriangulationFactory import TriangulationFactory




def RotateV1toV2(f, t):
    '''
    computes rotation matrix that transforms vector f to vector t
    
    :Args:
        - f: vector
        - t: vector
    
    :Returns: 
        - R: rotation matrix
    
    '''
    v = np.cross(f, t)  # cross product; row vector is required
    c = np.sum(f * t)  # dot product
    h = (1 - c) / (1 - c ** 2)

    R = np.array([[c + h * v[0] ** 2, h * v[0] * v[1] - v[2], h * v[0] * v[2] + v[1]],
                      [h * v[0] * v[1] + v[2], c + h * v[1] ** 2, h * v[1] * v[2] - v[0]],
                      [h * v[0] * v[2] - v[1], h * v[1] * v[2] + v[0], c + h * v[2] ** 2]])
    
    return R


def R_from_eigVec(eigVec):
    '''
    determine x,y,z axes and return rotation matrix
    
    :Args:
        - eigVec: eigenVectors matrix (sorted by eigenvalues - decreasing order)
    
    :Returns: 
        - R: rotation matrix
        
    ''' 
    
    #    determines which one is x/y axis
    if np.all(np.sign(np.cross(eigVec[:, 0], eigVec[:, 1])) == np.sign(eigVec[:, 2])):
        xind, yind = 0, 1
    else:
        xind, yind = 1, 0
                        
    R = np.hstack((np.array([eigVec[:, 0]]).T, np.array([eigVec[:, 1]]).T, np.array([eigVec[:, 2]]).T))
    return R


def PCA(pointArr):
    '''
    find eigVal, eigVec of covariance matrix and the order of eigenvalues in decreasing order
    
    :Args:
        - pointArr: pointset
    
    :Returns:
        - eigVal: eigenvalues
        - eigVec: eigenvectors
        
    '''
    covCell = np.cov(pointArr.T)  # covariance matrix of pointset
    eigVal, eigVec = np.linalg.eig(covCell)  # eigVal and eigVec of the covariance matrix  

    #   eigenvalues indices (x, y, z) 
    l1 = (np.where(np.abs(eigVec[0, :]) == np.max(abs(eigVec[0, :]))))[0][0]
    l2 = (np.where(np.abs(eigVec[1, :]) == np.max(abs(eigVec[1, :]))))[0][0]
    l3 = (np.where(np.abs(eigVec[2, :]) == np.max(abs(eigVec[2, :]))))[0][0]  # Z axis index in the eigVec matrix
    
    #    sort eigenvalues and eigenvectors
    eigVal = np.hstack((eigVal[l1], eigVal[l2], eigVal[l3]))
    eigVal[eigVal < 1e-10] = 0
    eigVec = np.hstack((np.array([eigVec[:, l1]]).T, np.array([eigVec[:, l2]]).T, np.array([eigVec[:, l3]]).T))
    
    return eigVal, eigVec


def Roughness(pointArr, rotM):
    '''
    roughness calculation
    
    :Args:
        - pointArr: points array(nx3)
        - rotM: rotation matrix
    
    :Returns: 
        - roughness
    
    '''
    
    C = np.mean(pointArr, 0)  # points mean
    pnt_cx = np.array([pointArr[:, 0] - C[0]]).T
    pnt_cy = np.array([pointArr[:, 1] - C[1]]).T
    pnt_cz = np.array([pointArr[:, 2] - C[2]]).T
    pntC = np.hstack((pnt_cx, pnt_cy, pnt_cz))  # points coordinates relatively to centroid
    pntR = np.dot(rotM, pntC.T).T 
    pnt_final = pntR + C
    roughness = np.std(pnt_final[:, 2])
    
    return roughness



def Points_in_window(cellLim, points):
    '''
    find points in defined window
    
    :Args:
        - cellLim: tuple of cell's limits coordinates (XminCell, XmaxCell, YminCell, YmaxCell)
        - points: pointset
        
    :Returns:
        - pointSet: subset of type PointSubSet from original pointset
    '''
    
    XminCell, XmaxCell, YminCell, YmaxCell = cellLim
    wind = Path([(XminCell, YmaxCell), (XmaxCell, YmaxCell), (XmaxCell, YminCell), (XminCell, YminCell)])
    wind.iter_segments()          
    pntInCell = wind.contains_points(np.hstack((np.array([points.X]).T, np.array([points.Y]).T)))
            
    indices = np.nonzero(pntInCell)
    pointSet = PointSubSet(points, np.array(indices)[0])
    
    return pointSet



def SmallGridCell(row, column, cellH, cellW, gridLim, points, rotM, fig):
    '''
    find points in the grid cell and cell's normal and roughness

    :Args:
        - row, column: number of rows and columns
        - cellH, cellW: cell's height and width
        - gridLim: [min X, max Y]
        - points: pointset in big grid cell 
        - rotM: transformation matrix to BigGrid normal system
    
    :Returns: 
        - cellCharact: cell's characteristics
            - CellLim: cells' limits [xmin, xmax, ymin, ymax](nx4)
            - CellN: cells' normals (nx3)
            - CellRough: cells' roughness 
            - CellNrel: normals of small cells relatively to normal of big cell
    
    '''
        
    #    ---------------- definition of matrices for results

    CellLim = np.zeros((row * column, 4))  # [xmin, xmax, ymin, ymax]
    CellN = np.zeros((row * column, 3))  # [nx, ny, nz]
    CellNrel = np.zeros((row * column, 3))  # [nx, ny, nz]
    CellRough = np.zeros((row * column, 1))  # cell's roughness
    
    #    ---------------- 
    
#    fig2 = mlab.figure(bgcolor=(1, 1, 1))
        
    xMin, yMax = gridLim[0], gridLim[1]

    indCell = 0
    
    for i in xrange(row):

        YmaxCell = yMax - cellH * i
        YminCell = yMax - cellH * (i + 1)
        
        for j in xrange(column):
            
            XmaxCell = xMin + cellW * (j + 1)
            XminCell = xMin + cellW * j
            
            #    ---------------- find points in cell
            pointSet = Points_in_window((XminCell, XmaxCell, YminCell, YmaxCell), points)
            
            CellLim[indCell, :] = np.hstack((XminCell, XmaxCell, YminCell, YmaxCell)) 

            #    ---------------- if a pointset is not empty do computations
            if pointSet.Size > 10:
                
                pointArr = pointSet.ToNumpy  # PointSet to numpy array 

                #    eigenVectors and eigenValues computation                                                           
                eigVal, eigVec = PCA(pointArr)  # eigVal and eigVec of the covariance matrix            

                #    find cell's characteristics  
                eigVal = np.sort(eigVal)[::-1]
                sigma = np.sqrt(eigVal)
                a1D = (sigma[0] - sigma[1]) / sigma[0]
                a2D = (sigma[1] - sigma[2]) / sigma[0]
                a3D = sigma[2] / sigma[0]
                
                denomP = np.sum(eigVal)
                p1 = eigVal[0] / denomP
                p2 = eigVal[1] / denomP
                p3 = eigVal[2] / denomP
                
                N = np.array([eigVec[:, 2]]).T
#                print N
                CellN[indCell, :] = N.T[0]
                CellNrel[indCell, :] = np.dot(rotM, N).T
                CellRough[indCell] = Roughness(pointArr, rotM)
                
#                fig = Visualization.RenderPointSet(pointSet, renderFlag='color',
#                                                    _figure=fig, color=(omnivar * 80, omnivar * 60, omnivar * 60), pointSize=2)
#                fig = Visualization.RenderPointSet(pointSet, renderFlag='color',
#                                                    _figure=fig, color=(CellRough[indCell][0] * 60, CellRough[indCell][0] * 60, CellRough[indCell][0] * 80), pointSize=2)
                    
                fig = Visualization.RenderPointSet(pointSet, renderFlag='color',
                                                    _figure=fig, color=(p1, p2, p3), pointSize=2)
#                fig = Visualization.RenderPointSet(pointSet, renderFlag='color',
#                                                    _figure=fig, color=(N[0][0], N[1][0], N[2][0]), pointSize=2)    
            else:
                CellN[indCell, :] = np.array([0, 0, 0])
                CellNrel[indCell, :] = np.array([0, 0, 0])
                CellRough[indCell] = 0
                     
            indCell += 1    
                
    #    cells characteristics to return             
    cellCharact = (CellLim, CellN, CellNrel, CellRough)  
                             
    #    Visualization.Show()      
    return cellCharact, fig   



def BigGridCell(row, column, gridLim, points, cellH, cellW, smallGrid_param, fig):
    '''
    find points in the grid cell and cell's normal and roughness

    :Args:
        - row, column: number of rows and columns
        - gridLim: [min X, max Y]
        - points: pointset
        - cellH, cellW: cell's height and width
        - smallGrid_param: number of rows and columns in the cell of big grid, cell's height and cell's width 
        
    :Returns: 
        - cellCharact: small cell's characteristics
            - CellLim: cells' limits [xmin, xmax, ymin, ymax](nx4)
            - CellN: cells' normals (nx3)
            - CellRough: cells' roughness (for small cells)
            - CellNrel: normals of small cells relatively to normal of big cell
    
    '''
        

    xMin, yMax = gridLim[0], gridLim[1]
    
    #    ---------------- definition of matrices for results

    
    CellLim = np.zeros((1, 4))  # [xmin, xmax, ymin, ymax]
    CellN = np.zeros((1, 3))  # [nx, ny, nz]
    CellNrel = np.zeros((1, 3))  # [nx, ny, nz]
    CellRough = np.zeros((1, 1))  # cell's roughness
    
    #    ---------------- 
     
    
    for i in xrange(row):

        YmaxCell = yMax - cellH * i
        YminCell = yMax - cellH * (i + 1)
        
        for j in xrange(column):
            
            XmaxCell = xMin + cellW * (j + 1)
            XminCell = xMin + cellW * j
            
            #    find points in cell
            pointSet = Points_in_window((XminCell, XmaxCell, YminCell, YmaxCell), points)
            
            #    ---------------- if a pointset is not empty do computations
            if pointSet.Size > 10:
                color = np.random.random(3)
#                fig1 = Visualization.RenderPointSet(pointSet, renderFlag='color', _figure=fig1, color=(color[0], color[1], color[2]), pointSize=2)
                
                pointArr = pointSet.ToNumpy  # PointSet to numpy array 

                #    eigenVectors and eigenValues computation                               
                eigVal, eigVec = PCA(pointArr)  # eigVal and eigVec of the covariance matrix  
           
                
                #    return rotation matrix to BigGrid cell normal
                CellR = R_from_eigVec(eigVec)
                rS, cS, cH, cW = smallGrid_param
                rotMat = np.reshape(CellR, (3, 3))

                smallGrid, fig = SmallGridCell(rS, cS, cH, cW, (XminCell, YmaxCell), pointSet, rotMat, fig)

                #    cells characteristics  
                CellLim = np.vstack((CellLim, smallGrid[0]))
                CellN = np.vstack((CellN, smallGrid[1]))
                CellNrel = np.vstack((CellNrel, smallGrid[2]))
                CellRough = np.vstack((CellRough, smallGrid[3]))


    #    cells characteristics to return            
    cellCharact = (CellLim[1:, :], CellN[1:, :], CellNrel[1:, :], CellRough[1:])  
                                    
#    Visualization.Show()      
    return cellCharact     
  



def GridNormal(pnt, overlap, cellW, cellH, fig):
    '''
    compute relative normals of the grid's cells 
    (normal value relatively to dominant normal) 
    
    :Args: 
        - pnt: point set
        - overlap: percentage of overlap
        - cellW, cellH: small grid cell dimensions
    
    :Returns: 
        - smallGridLim: limits of every cell (min, max coord.)
        - smallGridNrel: normals of small cells relatively to normal of big cell
        - smallGridN: cells' normals (nx3)
        - smallGridRough: cells' roughness

    '''
#    boundaries of a pointset
    xMin, xMax, yMin, yMax = np.min(pnt.X), np.max(pnt.X), np.min(pnt.Y), np.max(pnt.Y)
    
#    grid width and height
    w = xMax - xMin
    h = yMax - yMin
    
#    divide the area in big cells according to overlap area (%) 
    wReg = np.sqrt(overlap) * w
    hReg = np.sqrt(overlap) * h
    
#    number of rows and columns in big grid
    nW = np.int32(np.ceil(w / wReg))
    nH = np.int32(np.ceil(h / hReg))
#    new width and height
    w = nW * wReg
    h = nH * hReg
    
#    number of rows and columns in small grid
    row = np.int32(np.ceil(hReg / cellH))
    column = np.int32(np.ceil(wReg / cellW)) 
    
    cellH = hReg / row
    cellW = wReg / column
    
#    big grid cells' limits and normals calculation
    cellCharact = BigGridCell(nH, nW, np.array([xMin, yMax]), pnt, hReg, wReg, (row, column, cellH, cellW), fig)  
    
    smallGridLim, smallGridN, smallGridNrel, smallGridRough = cellCharact
    
#    delete empty cells
    rcZeroes = np.where(smallGridN == np.zeros((1, 3)))
    smallGridLim = np.delete(smallGridLim, (rcZeroes[0])[::3], 0)
    smallGridN = np.delete(smallGridN, (rcZeroes[0])[::3], 0)
#    print smallGridN
    smallGridNrel = np.delete(smallGridNrel, (rcZeroes[0])[::3], 0)
    smallGridRough = np.delete(smallGridRough, (rcZeroes[0])[::3], 0)
    
    return smallGridLim, smallGridNrel, smallGridN, smallGridRough
    




if __name__ == '__main__':
    # Load pointset
    print "Loading data..."         
    fileName1 = 'D:\\Documents\\Pointsets\\Bonim_set1.pts'  
    fileName2 = 'D:\\Documents\\Pointsets\\Bonim_set2.pts' 
    
    print "Preparing data..."
    points1 = Registration.PointSet2Array(fileName1)
    points2 = Registration.PointSet2Array(fileName2)
    
#    points1 = points1[::5, :]
#    points2 = points2[::5, :]  
#    pnt1 = Registration.TransformData(points1, np.array([[0], [0], [0]]), -90, np.array([1, 0, 0]))
#    pnt2 = Registration.TransformData(points2, np.array([[0], [0], [0]]), 90, np.array([0, 1, 0]))
#    pnt1 = PointSet(pnt1)
#    pnt2 = PointSet(pnt2)
    
    pnt1 = PointSet(points1[::10, :])
    pnt2 = PointSet(points2[::10, :])
    
    tp = TriangulationFactory.Delaunay2D(pnt1)
    tp.TrimEdgesByLength(0.7)
    Visualization.RenderTriangularMesh(tp, renderFlag='height')
    Visualization.Show()
    
    del points1, points2

    print "Creating grid..."    
    n = 5  # cell height
    m = 5  # cell width
    overlap = 0.3  # point clouds estimated overlap (%)
    
    fig1 = mlab.figure(bgcolor=(1, 1, 1))  
    smallGridLim1, smallGridNrel1, smallGridN1, smallGridRough1 = GridNormal(pnt1, overlap, m, n, fig1)
    
    fig2 = mlab.figure(bgcolor=(1, 1, 1))
    smallGridLim2, smallGridNrel2, smallGridN2, smallGridRough2 = GridNormal(pnt2, overlap, m, n, fig2)
    
    print 'smallGridN1 = ', '\r', smallGridNrel1
    print 'smallGridN2 = ', '\r', smallGridNrel2
    print 'smallGridRough1 = ', '\r', smallGridRough1
    print 'smallGridRough2 = ', '\r', smallGridRough2
      
    Visualization.Show()

   

 
