'''
infragit
reuma\Reuma
08, May, 2018 
'''

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

import numpy as np
from PointSet import PointSet
from ColorProperty import ColorProperty

if __name__ == '__main__':
    pts = PointSet(np.array([[1, 1, 1], [2, 1, 1], [3, 1, 1]]))
    colors = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    color = ColorProperty(pts, colors)
    # b = BaseProperty(pts)
    # b.save(filename = r'savedProperty\temp.h5')
    color.save(filename = r'savedProperty\color.h5')
