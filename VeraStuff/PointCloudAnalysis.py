'''
Created on 30 march 2014

@author: vera
'''

import numpy as np
import numpy.matlib as matlib
from Registration import Registration

from IOFactory import IOFactory
from NormalsFactory import NormalsFactory


def PointSet2Array_Norm(fileName):
        '''
        reads pointLists, convert in to ndarray and compute normals 
        input: points file name
        output: array: [pnt; norm]
        
        '''
        pointSetList = []
        vert = []
        n = IOFactory.ReadPts(fileName, pointSetList, vert)  # returns number of pointsets
        Norm = (NormalsFactory.VtkNormals(pointSetList[0])).Normals()  # computes normals of each point 
#        creates an array [pnt; norm]
        Pnt = pointSetList[0].ToNumpy()
        
        for i in range(n):
            if i != 0:
                tempP = Pnt
                data = pointSetList[i].ToNumpy()
                Pnt = np.vstack((tempP, data))
                tempN = Norm
                norm_data = (NormalsFactory.VtkNormals(pointSetList[i])).Normals()
                Norm = np.vstack((tempN, norm_data))        
#        pointSet_Res = PointSet(sPnt) 
#        fig1 = Visualization.RenderPointSet(pointSet_Res, renderFlag='color', color=(1, 0, 0))
#        pointSet_Target = PointSet(targetPnt)
#        fig1 = Visualization.RenderPointSet(pointSet_Target, renderFlag='color', _figure=fig1, color=(0, 0, 1))
#        Visualization.Show()

        return np.hstack((Pnt, Norm))
    
    
def GridCalc(row, column, gridLim, points_norm, cellH, cellW):
    '''
    finds points in the grid cell
    
    '''
    
    print "Cells' normals calculation..."

    xMin = gridLim[0], xMax = gridLim[1], yMin = gridLim[2], yMax = gridLim[3]
    
    pointInd_Cell = np.empty((row, column, 1), dtype=object)
    indN = -1
    
    for i in xrange(row):
        num = 0
#        find points in 'y' boundaries
        P1inCell_Y = np.where((points_norm[:, 1] <= yMax - cellH * i) & (points_norm[:, 1] >= yMax - cellH * (i + 1)))
        if P1inCell_Y.size != 0:
            for j in xrange(column): 
                indN += 1 
#                find points from 'y' boundaries in 'x' boundaries  
                P1inCell = np.where((points_norm[P1inCell_Y, 0] >= xMin + cellW * j) & (points_norm[P1inCell_Y, 0] <= xMin + cellW * (j + 1)))
                pointInd_Cell[i, j, 1] = P1inCell_Y[P1inCell]
                
    return pointInd_Cell





if __name__ == '__main__':
    
    # Load pointset
    print "Loading data..."         
    fileName1 = 'D:\\My Documents\\Pointsets\\r1.pts'  
    fileName2 = 'D:\\My Documents\\Pointsets\\r2.pts' 
    
    print "Preparing data..."
    points_norm1 = PointSet2Array_Norm(fileName1)
#    print points_norm1
    points_norm2 = PointSet2Array_Norm(fileName2)

    points_norm1 = points_norm1[::5, :]
    points_norm2 = points_norm2[::5, :]  
    pnt1 = Registration.TransformData(points_norm1[:, 0:3], np.array([[0], [0], [0]]), -90, np.array([1, 0, 0]))
    pnt2 = Registration.TransformData(points_norm2[:, 0:3], np.array([[0], [0], [0]]), 90, np.array([0, 1, 0]))
    points_norm1[:, 0:3] = pnt1
    points_norm2[:, 0:3] = pnt2

#    =========================================================================
    
#    print 'Drawing points...'
#    points2displ1 = PointSet(points_norm1[:, 0:3])
#    fig = Visualization.RenderPointSet(points2displ1, renderFlag='color', color=(1, 0, 0), pointSize=3.0) 
#    points2displ2 = PointSet(points_norm2[:, 0:3])
#    fig = Visualization.RenderPointSet(points2displ2, renderFlag='color', _figure=fig, color=(0, 1, 0), pointSize=3.0) 
#    Visualization.Show()


#    =========================================================================

    print "Creating grid..."    
    n = 0.2  # cell height
    m = 0.2  # cell width
    
#    boundaries of a grid
    xMin1 = np.min(points_norm1[:, 0])
    xMax1 = np.max(points_norm1[:, 0])
    yMin1 = np.min(points_norm1[:, 1])
    yMax1 = np.max(points_norm1[:, 1])
    
#    grid width and height
    w1 = xMax1 - xMin1
    h1 = yMax1 - yMin1
    
#    divide the area in big cells according to overlap area (%) 
    wReg1 = np.floor(np.sqrt(0.3) * w1)  # overlap area is 30%
    hReg1 = np.floor(np.sqrt(0.3) * h1)
    
    nW1 = np.int32(np.ceil(w1 / wReg1))
    nH1 = np.int32(np.ceil(h1 / hReg1))
    w1 = nW1 * wReg1
    h1 = nH1 * hReg1
    
    BigGrid1 = np.empty((nH1, nW1, 1), dtype=object)
       
#    number of columns and rows in the grid
    row1 = np.int32(np.ceil(h1 / n))
    column1 = np.int32(np.ceil(w1 / m))    
    
    
#    boundaries of a grid
    xMin2 = np.min(points_norm2[:, 0])
    xMax2 = np.max(points_norm2[:, 0])
    yMin2 = np.min(points_norm2[:, 1])
    yMax2 = np.max(points_norm2[:, 1])
    
    w2 = xMax2 - xMin2
    h2 = yMax2 - yMin2
    
#    divide the area in big cells according to overlap area (%) 
    wReg2 = np.floor(np.sqrt(0.3) * w2)  # overlap area is 30%
    hReg2 = np.floor(np.sqrt(0.3) * h2)
    
    nW2 = np.int32(np.ceil(w2 / wReg2))
    nH2 = np.int32(np.ceil(h2 / hReg2))
    w2 = nW2 * wReg2
    h2 = nH2 * hReg2
    
    BigGrid2 = np.empty((nH2, nW2, 1), dtype=object)
    
#    number of columns and rows in the grid
    row2 = np.int32(np.ceil(h2 / n))
    column2 = np.int32(np.ceil(w2 / m))
    
    
    zAxis = np.array([0, 0, 1])


#    =========================================================================
    print "Grid 1..."
     

    print "Cells' normals calculation..."

    cellNorm1 = np.zeros((row1 * column1, 5))  # cellNorm = [numRow, numCol, norm(1x3)]
    cellNormAng1 = np.zeros((row1 * column1, 1))
    indN = -1
    for i in xrange(row1):
        num = 0
#        find points in 'y' boundaries
        P1inCell_Y = points_norm1[(points_norm1[:, 1] <= yMax1 - n * i) & (points_norm1[:, 1] >= yMax1 - n * (i + 1))]
        if P1inCell_Y.size != 0:
            for j in xrange(column1): 
                indN += 1 
#                find points from 'y' boundaries in 'x' boundaries  
                P1inCell = P1inCell_Y[(P1inCell_Y[:, 0] >= xMin1 + m * j) & (P1inCell_Y[:, 0] <= xMin1 + m * (j + 1))]
                if P1inCell.size != 0:
#                    cell's center
                    xC = (xMin1 + m * j + xMin1 + m * (j + 1)) / 2
                    yC = (yMax1 - n * i + yMax1 - n * (i + 1)) / 2
                    
                    distC = np.sqrt((P1inCell[:, 0] - xC) ** 2 + (P1inCell[:, 1] - yC) ** 2)  # dist of every point to cell's center
                    w = 1 / distC  # point's normal weight
                    wN = np.array([w / np.sum(w)])
                    cellNtemp = np.mean(np.dot(wN, P1inCell[:, 3:6]), axis=0)
                    cN = cellNtemp / np.linalg.norm(cellNtemp)
                    cellNorm1[indN, :] = np.hstack((i, j, cN))
                    
                    normAng = np.arccos(np.sum(cN * zAxis))
                    if normAng > np.pi / 2:
                        normAng = np.pi - normAng
                    cellNormAng1[indN] = normAng

    print cellNormAng1
    

    print "Grid 2..."
    print "Dominant normal calculation..."
    
    print "Cells' normals calculation..."
    cellNorm2 = np.zeros((row2 * column2, 5))  # cellNorm = [numRow, numCol, norm(1x3)]
    cellNormAng2 = np.zeros((row2 * column2, 1))
    indN = -1
    for i in xrange(row2):
        num = 0
        P1inCell_Y = points_norm2[(points_norm2[:, 1] <= yMax2 - n * i) & (points_norm2[:, 1] >= yMax2 - n * (i + 1))]
        if P1inCell_Y.size != 0:
            for j in xrange(column2):    
                P1inCell = P1inCell_Y[(P1inCell_Y[:, 0] >= xMin2 + m * j) & (P1inCell_Y[:, 0] <= xMin2 + m * (j + 1))]
                indN += 1
                if P1inCell.size != 0:
#                    cell's center
                    xC = (xMin2 + m * j + xMin2 + m * (j + 1)) / 2
                    yC = (yMax2 - n * i + yMax2 - n * (i + 1)) / 2
                    
                    distC = np.sqrt((P1inCell[:, 0] - xC) ** 2 + (P1inCell[:, 1] - yC) ** 2)  # dist of every point to cell's center
                    w = 1 / distC  # point's normal weight
                    wN = np.array([w / np.sum(w)])
                    cellNtemp = np.mean(np.dot(wN, P1inCell[:, 3:6]), axis=0)
                    cN = cellNtemp / np.linalg.norm(cellNtemp)
                    cellNorm2[indN, :] = np.hstack((i, j, cN))
                    
                    normAng = np.arccos(np.sum(cN * zAxis))
                    if normAng > np.pi / 2:
                        normAng = np.pi - normAng
                    cellNormAng2[indN] = normAng
                    
    print cellNormAng2
    
    NA1 = matlib.repmat(cellNormAng1.T, cellNormAng2.size, 1)
    NA2 = matlib.repmat(cellNormAng2, 1, cellNormAng1.size)
    delta = NA1 - NA2  # differences between normals of pntset1 and normals of pntset2
#    similarity = np.where(np.abs(delta) == np.min(np.abs(delta), 0))  # find the minimum difference
#    similarity = np.array(similarity)
#    print delta
    similarity = np.where(np.abs(delta) < (3 * np.pi / (60 * 180)))  
#    similarity_new = similarity[:, simil[0]]
    print similarity
    
#    similar angles
    simil_na1 = cellNormAng1[similarity[1]]
    simil_na2 = cellNormAng1[similarity[0]]
    print 'Norm1 = ', simil_na1, '\r', 'Norm2 = ', simil_na2
    
#    similar normals
    simil_N1 = cellNorm1[similarity[1], 2:5]
    simil_N2 = cellNorm1[similarity[0], 2:5]
#    print 'Norm1 = ', simil_N1, '\r', 'Norm2 = ', simil_N2
#    
#    A = np.zeros((simil_N1.shape[0] * 3, 9))
#    A[::3, 0:3] = simil_N1
#    A[1::3, 3:6] = simil_N1
#    A[2::3, 6:10] = simil_N1
#    L = np.reshape(simil_N2, ((simil_N2.shape[0] * 3, 1)), 0)
#    xR = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, L))
# #    print 'xR = ', xR
#    
#    R = np.hstack((xR[0:3], xR[3:6], xR[6:9]))
#    print R
#    pnt2t = (np.dot(R.T, points_norm2[:, 0:3].T)).T
    
# #    print 'Drawing points...'
#    points2displ1 = PointSet(points_norm2[:, 0:3])
#    fig = Visualization.RenderPointSet(points2displ1, renderFlag='color', color=(1, 0, 0), pointSize=3.0) 
#    points2displ2 = PointSet(pnt2t)
#    fig = Visualization.RenderPointSet(points2displ2, renderFlag='color', _figure=fig, color=(0, 1, 0), pointSize=3.0)
#    points2displ3 = PointSet(points_norm1[:, 0:3]) 
#    fig = Visualization.RenderPointSet(points2displ3, renderFlag='color', _figure=fig, color=(0, 0, 1), pointSize=3.0)
#    Visualization.Show()
