'''
infraGit
photo-lab-3\Reuma
16, Jan, 2017 
'''

import numpy as np
from matplotlib import pyplot as plt
from BaseProperty import BaseProperty
import platform

if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('TkAgg')


class RasterProperty( BaseProperty ):
    """
    A raster representation of the point set
    """
    __dataType = ""  # A string indicating the type of data stored in the panoramic view (e.g. range, intensity, etc.)
    __rasterData = None  # A m-by-n-by-p array in which the raster is stored
    __voidData = -9999  # A number indicating missing data in the raster
    __cellSize = 0.05  # The spacing between grid cells
    __ncolumns = None
    __nrows = None

    #Grid bounding box
    __extent = None #two corners of the bounding box; dictionary.

    def __init__(self, points, data, gridSpacing, **kwargs ):
        """
        constructor
        :param points: PointSet object from which the panorama will be created
        :param gridSpacing: The spacing between grid cells
        :param extent: the spatial extent of the raster
        :param ncolumns: number of columns in the raster
        :param nrows: number of rows in raster
        """

        self.__cellSize = gridSpacing
        self.__points = points
        self.__rasterData = data

        if ('dataType' in kwargs.keys()):
            self.__dataType = kwargs['dataType']
        if ('extent' in kwargs.keys()):
            self.__extent = kwargs['extent']
        if ('ncolumns' in kwargs.keys()):
            self.__ncolumns = kwargs['ncolumns']
        else:
            self.__ncolumns = data.shape[1]
        if ('nrows' in kwargs.keys()):
            self.__nrows= kwargs['nrows']
        else:
            self.__nrows = data.shape[0]

    @classmethod
    def setValues(self, **kwargs):
        """
        sets self values
        :param in kwargs: extent, ncolumns, nrows, cellSize, noDataValue,

        """
        if ('extent' in kwargs.keys()):
            self.__extent = kwargs['extent']

        if ('ncolumns' in kwargs.keys()):
            self.__ncolumns = kwargs['ncolumns']
        if ('nrows' in kwargs.keys()):
            self.__nrows= kwargs['nrows']

        if ('cellSize' in kwargs.keys()):
            self.__cellSize = kwargs['cellSize']

        if ('noDataValue' in kwargs.keys()):
            self.__cellSize = kwargs['noDataValue']


if __name__ == '__main__':
    pass