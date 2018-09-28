'''
infragit
reuma\Reuma
08, May, 2018 
'''

import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')

from RasterData import RasterData
from ColorProperty import ColorProperty
from IOFactory import IOFactory
from PointSet import PointSet
import numpy as np

if __name__ == '__main__':
    pts = PointSet(np.array([[1, 1, 1], [2, 1, 1], [3, 1, 1]]))
    raster = IOFactory.rasterFromAscFile(r'D:\ownCloud\Data\tt3.txt')
    raster.setValues(points = pts)
    raster.save(r'..\ReumaPhD\savedProperty\raster6.h5')
    # colors = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    # color = ColorProperty(pts, colors)
    # IOFactory.saveProperty(r'savedProperty\color4.h5', color)
    # b.save(filename = r'savedProperty\temp.h5')
    raster = IOFactory.load(r'savedProperty\raster6.h5', RasterData)
    pts2 = IOFactory.load(r'savedProperty\color4.h5', ColorProperty)
    print('hello')
