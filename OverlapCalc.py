'''
Created on 28 march 2014

@author: vera
'''
import numpy as np

from Registration import Registration

if __name__ == '__main__':
    FileNames = ['D:\\Documents\\Pointsets\\Ahziv\\st1.pts', 'D:\\Documents\\Pointsets\\Ahziv\\st2.pts', 'D:\\Documents\\Pointsets\\Ahziv\\st3.pts',
                 'D:\\Documents\\Pointsets\\Ahziv\\st4.pts', 'D:\\Documents\\Pointsets\\Ahziv\\st5.pts']
    for file1 in xrange(len(FileNames) - 1):
        for file2 in xrange(file1 + 1, len(FileNames)):
            # Load 2 pointsets
#             print "Loading data..."     
            fileName1 = FileNames[file1]
            fileName2 = FileNames[file2]
            
#             # approximate registration values 
#             transl1 = (np.array([0, 0, 0])).T
#             axis1 = np.array([0, 0, 1])
#             angle1 = 0
#             
#             transl2 = (np.array([0, 0, 0])).T
#             axis2 = np.array([0, 0, 1])
#             angle2 = 0
            
#             print "Preparing data..."
#             points1 = Registration.PointSet2Array(fileName1)
#             points2 = Registration.PointSet2Array(fileName2)
            pointset1 = Registration.PointSet2Array(fileName1)
            pointset2 = Registration.PointSet2Array(fileName2)
            
#             pointset1 = Registration.TransformData(points1, transl1, angle1, axis1)
#             pointset2 = Registration.TransformData(points2, transl2, angle2, axis2)
#             del points1, points2
        #    pointset1 = pointset1[::10, :]
        #    pointset2 = pointset2[::10, :]
            var1 = np.var(pointset1[:, 2], 0)
            var2 = np.var(pointset2[:, 2], 0)
            
        #    pnt12disp = PointSet(pointset1) 
        #    fig = Visualization.RenderPointSet(pnt12disp, renderFlag='color', color=(1, 0, 0), pointSize=3.0)
        #    pnt22disp = PointSet(pointset2)
        #    fig = Visualization.RenderPointSet(pnt22disp, renderFlag='color', _figure=fig, color=(0, 0, 1), pointSize=3.0)
        #    Visualization.Show()
            
            
        #    overlap calculation
        
            points = np.vstack((pointset1, pointset2))
#             print "Creating grid..."
        #    boundaries of a grid
            xMin = np.min(points[:, 0])
            xMax = np.max(points[:, 0])
            yMin = np.min(points[:, 1])
            yMax = np.max(points[:, 1])
            del points
            
            w = xMax - xMin
            h = yMax - yMin
            
            n = 10  # cell height
            m = 10  # cell width
            row = np.int32(np.ceil(h / n))
            column = np.int32(np.ceil(w / m))
            area = (column * n) * (row * m)
            
        #    print xMin, xMax, yMin, yMax
#             print "Overlap calculation..."
            count = 0
            for i in xrange(row):
                num = 0
                P1inCell_Y = pointset1[(pointset1[:, 1] <= yMax - n * i) & (pointset1[:, 1] >= yMax - n * (i + 1))]
                P2inCell_Y = pointset2[(pointset2[:, 1] <= yMax - n * i) & (pointset2[:, 1] >= yMax - n * (i + 1))]
                if P1inCell_Y.size != 0 and P2inCell_Y.size != 0:
                    for j in xrange(column): 
                        if num == np.minimum(P1inCell_Y.shape[0], P2inCell_Y.shape[0]):
                            break   
                        P1inCell = P1inCell_Y[(P1inCell_Y[:, 0] >= xMin + m * j) & (P1inCell_Y[:, 0] <= xMin + m * (j + 1))]
                        P2inCell = P2inCell_Y[(P2inCell_Y[:, 0] >= xMin + m * j) & (P2inCell_Y[:, 0] <= xMin + m * (j + 1))]
                        if P1inCell.size != 0 and P2inCell.size != 0:
                            count += (n * m)
                            num += 1
                        else:
                            if P1inCell.size == 0 and P2inCell.size == 0: 
                                area -= (n * m)
                else:
                    if P1inCell_Y.size == 0 and P2inCell_Y.size == 0: 
                        area -= column * (n * m)
            
#             print area     
            print "stations", " ", file1 + 1, " and ", file2 + 1, ":\t", count * 100 / area  # percent of overlap
            
            
        
