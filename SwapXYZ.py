import numpy as np
import vtk as vtk
from IOFactory import IOFactory
from PointSet import PointSet
from Visualization import Visualization 
from Registration import Registration


if __name__ == '__main__':
    fileName_s = 'D:\\My Documents\\Ovda\\Ovda5.pts'
    sData = Registration.PointSet2Array(fileName_s)
    sData = np.hstack((np.array([sData[:, 0]]).T, np.array([sData[:, 2]]).T, np.array([sData[:, 1]]).T))
    
    print "file"
    f = open('D:\\My Documents\\Ovda\\Ovda5_Swap.txt', 'w')
    func = lambda v: (f.write(str(v[0]) + '\t' + str(v[1]) + '\t' + str(v[2]) + '\n'))
    points = np.char.mod('%08f', sData)
    map(func, sData)
        
    f.close()
    
    print "done"
