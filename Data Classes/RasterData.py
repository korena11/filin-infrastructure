'''
infraGit
photo-lab-3\Reuma
16, Jan, 2017 
'''

import platform

from numpy import array, floor, nonzero, logical_or, logical_and, logical_not, zeros
from numpy import max as npmax
from pyproj import Proj, transform

from BaseData import BaseData
from PointSet import PointSet

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')


class RasterData(BaseData):
    """
    A raster representation of the point set and manipulations on it

    """
    __voidData = -9999  # A number indicating missing data in the raster
    __cellSize = 0.05  # The spacing between grid cells
    __spatialReference = None # spatial reference info

    # transformation to convert pixel to world coordinates
    __geoTransform = (0, 0, 1, 1)

    #Lidar data info
    __dataOrigin = "" # A string indicating the point acquiring means: 'terrestrial',
                # 'airborne' , 'mobile', 'slam', 'kinect', 'photogrammetry'
    __dataFeature = ""  # A string indicating the type of data stored  (e.g. range, intensity, etc.)
    __measurement_accuracy= 0.15 # The accuracy of a depth/height/range measurement
    __mean_roughness = .02 # roughness estimation of the raster data

    def __init__(self, data, gridSpacing, **kwargs):
        """
        constructor

        :param data: the rasterized data
        :param gridSpacing: The spacing between grid cells

        **Optionals**

        :param voidData: A number indicating missing data in the raster
        :param geoTrnasform: The transformation needed to convert from pixel to world coordinates (tx, ty, sx, sx)
        :param spatialReference: spatial reference info
        :param dataOrigin: A string indicating the point acquiring means: 'terrestrial',
        :param dataFeature: A string indicating the type of data stored  (e.g. range, intensity, etc.)
        :param accuracy: measurement accuracy estimation
        :param roughness: an estimation of the surface roughness
        :param path: the path for the datafile

        :type voidData: int
        :type geoTrnasform: tuple
        :type spatialReference: ??
        :type dataOrigin: str
        :type dataFeature: str
        :type accuracy: float
        :type roughness: float
        :type path: str

        """
        super(RasterData, self).__init__()
        self.__cellSize = gridSpacing
        self.setdata(data)

        self.setValues(**kwargs)

    def setValues(self, **kwargs):
        """
        sets self values

        :param voidData: A number indicating missing data in the raster
        :param geoTrnasform: The transformation needed to convert from pixel to world coordinates (tx, ty, sx, sx)
        :param spatialReference: spatial reference info
        :param dataOrigin: A string indicating the point acquiring means: 'terrestrial',
        :param dataFeature: A string indicating the type of data stored  (e.g. range, intensity, etc.)
        :param accuracy: measurement accuracy estimation
        :param roughness: an estimation of the surface roughness
        :param path: the path for the datafile

        :type voidData: int
        :type geoTrnasform: tuple
        :type spatialReference: ??
        :type dataOrigin: str
        :type dataFeature: str
        :type accuracy: float
        :type roughness: float
        :type path: str

        """
        params = {'voidData': self.__voidData,
                  'geoTransform': (self.__geoTransform),
                  'spatialReference': self.__spatialReference,
                  'cellSize': self.__cellSize,
                  'dataOrigin': self.__dataOrigin,
                  'dataFeature': self.__dataFeature,
                  'accuracy': self.__measurement_accuracy,
                  'roughness': self.__mean_roughness,
                  'path': self.path
                  }
        params.update(kwargs)
        self.__geoTransform = params['geoTransform']
        self.__spatialReference = params['spatialReference']
        self.setPath(params['path'])

        self.__voidData = params['voidData']
        self.__cellSize = params['cellSize'],
        self.__dataOrigin = params['dataOrigin']
        self.__dataFeature = params['dataFeature']
        self.__measurement_accuracy = params['accuracy']
        self.__mean_roughness = params['roughness']


    def pixelToWorld(self, row, col):
        """
        Transform a pixel to world coordinate system

        :param row: the pixel's row
        :param col: the pixel's column

        :type row: int
        :type col: int

        :return: x, y in world coordinates
        :rtype: tuple

        """
        x = self.scaleX * col + self.translationX
        y = self.scaleY * row + self.translationY
        return x, y

    def worldToPixel(self, x, y, spatialRef = None):
        """
        Conversion of a point from world coordinates to pixel ones

        :param x: The X-axis coordinate of the point to convert
        :param y: The Y-axis coordinate of the point to convert
        :param spatialRef: The spatial reference of the point to convert given in EPSG code

        :type x: float
        :type y: float
        :type spatialRef: ESPG code

        :return: the row and column that correspond to the given point (tuple of two float numbers)
        :rtype: tuple (int, int)

        """
        if not(spatialRef is None) and spatialRef != self.__spatialReference:
            rasterProj = Proj(init='epsg:' + self.__spatialReference)
            pointProj = Proj(init='epsg:' + spatialRef)
            x, y = transform(pointProj, rasterProj, x, y)

        row = (y - self.translationY) / self.scaleY
        col = (x - self.translationX) / self.scaleX

        return row, col

    def interpolateAtPoint(self, x, y, spatialRef=None, method='bilinear'):
        """
        Computing the height(s) of a given point(s) based on a given method

        :param x: x-coordinate(s) of the point(s)
        :param y: y-coordinate(s) of the point(s)
        :param spatialRef: The spatial reference of the point(s), given as an ESPG code
        :param method: The method to be used for the interpolation:
                        - 'nearest' - for nearest neighbor
                        - 'bilinear' - for bilinear intepolation (default)
                        - 'bicubic' - for bicubic interpolation (to be implemented)

        :return:

        """
        row, col = self.worldToPixel(x, y, spatialRef)
        return self.intepolateAtPixel(row, col, method)

    def intepolateAtPixel(self, row, col, method='bilinear'):
        """
        Computing the height(s) of a given pixel(s) based on a given method

        :param row: The row(s) of the pixel(s)
        :param col: The column(s) of the pixel(s)
        :param method: The method to be used for the interpolation (same as methods as in interpolateAtPoint method)

        :return:

        """
        from numpy import round, int_
        row = array([row]).reshape((-1, ))
        col = array([col]).reshape((-1, ))

        if row.shape != col.shape:
            return None

        values = zeros(row.shape)
        outOfBoundsTest = logical_or(row < 0, logical_or(row >= self.data.shape[0],
                                                         logical_or(col < 0, col >= self.data.shape[1])))
        values[nonzero(outOfBoundsTest)[0]] = None

        exactPointsTest = logical_and(abs(row - round(row)) < 1e-8, abs(col - round(col)) < 1e-8)
        exactPointsIdx = nonzero(exactPointsTest)[0]
        if len(exactPointsIdx) > 0:
            values[exactPointsIdx] = self.data[row[exactPointsIdx], col[exactPointsIdx]]

        otherPoints = nonzero(logical_and(logical_not(outOfBoundsTest),
                                          logical_not(exactPointsTest)))[0]

        if len(otherPoints) > 0:
            if method == 'nearest':
                return self.data[round(row[otherPoints]), round(col[otherPoints])]
            elif method == 'bilinear':
                try:
                    intRow, intCol = int_(floor(row[otherPoints])), int_(floor(col[otherPoints]))
                    z00 = self.data[intRow, intCol]
                    z10 = self.data[intRow, intCol + 1]
                    z01 = self.data[intRow + 1, intCol]
                    z11 = self.data[intRow + 1, intCol + 1]

                    a = z00
                    b = (z10 - z00)# / self.__scaleX
                    c = (z01 - z00)# / self.__scaleY
                    d = (z00 - z01 - z10 + z11)# / (self.__scaleX * self.__scaleY)

                    values[otherPoints] = a + b * (col - intCol) + c * (row - intRow) + d * (col - intCol) * \
                                                                                        (row - intRow)
                except:
                    print row, col, intRow, intCol
                    return None
            # elif method == 'bicubic':
            #     intRow, intCol = floor(row), floor(col)
            #     if intRow == 0 or intCol == 0 or intRow == self.__rasterData.shape[0] - 2 or \
            #                     intCol == self.__rasterData.shape[1] - 2:
            #         return self.interpolateAtPoint(x, y, spatialRef, 'bilinear')
            else:
                values[otherPoints] = None
        return values

    def getGtidPoints(self, xmin=None, ymin=None, xmax=None, ymax=None, spatialRef=None):
        from numpy import arange, meshgrid, ceil
        if not (xmin is None or ymin is None or xmax is None or ymax is None):
            rows, cols = self.worldToPixel(array([xmin, xmax]), array([ymin, ymax]), spatialRef)
            xv = arange(ceil(cols[0]), floor(cols[1]), dtype=int)
            yv = arange(ceil(rows[1]), floor(rows[0]), dtype=int)
        else:
            xv = arange(self.data.shape[1])
            yv = arange(self.data.shape[0])

        xs, ys = meshgrid(xv, yv)
        xs = xs.reshape((-1, ))
        ys = ys.reshape((-1, ))
        rasterData = self.data[xs, ys].reshape((-1,))

        xs = (self.translationX + self.scaleX * xs)
        ys = (self.translationY + self.scaleY * ys)

        if not (spatialRef is None) and spatialRef != self.__spatialReference:
            rasterProj = Proj(init='epsg:' + self.__spatialReference)
            pointProj = Proj(init='epsg:' + spatialRef)
            xs, ys = transform(rasterProj, pointProj, xs, ys)

        return array([xs, ys, rasterData]).T

#------------------ PROPERTIES --------------------------
    @property
    def spatialReference(self):
        """
        The spatial reference information

        """
        return self.__spatialReference

    @property
    def translationX(self):
        """
        The east translation

        """
        return self.__geoTransform[0]

    @property
    def translationY(self):
        """
        The north translation

        """
        return self.__geoTransform[1]

    @property
    def scaleX(self):
        return self.__geoTransform[2]

    @property
    def scaleY(self):
        return self.__geoTransform[3]

    @property
    def shape(self):
        """
        Raster's rows and columns size

        """
        return self.data.shape

    @property
    def rows(self):
        """
        Number of rows in the raster

        """
        return self.data.shape[0]

    @property
    def cols(self):
        """
        Number of columns in the raster

        """
        return self.data.shape[1]

    @property
    def resolution(self):
        """
        Raster resolution
        """
        return self.__cellSize

    @property
    def accuracy(self):
        """
        Measurement accuracy

        """
        return self.__measurement_accuracy

    def roughness(self, **kwargs):
        """
        An estimation of the surface roughness

        """
        if kwargs:
            self.__mean_roughness = kwargs['roughness']
        else:
            return self.__mean_roughness

    @property
    def min_range(self):
        """
        The closest range in the data

        """
        return npmax(npmax(self.data))

    def ToPointSet(self):
        """
        Transforms raster to PointSet data

        :return: PointSet of the raster data

        :rtype: PointSet

        """
        # pts = PointSet()




if __name__ == '__main__':
    pass
