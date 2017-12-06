'''
infraGit
photo-lab-3\Reuma
16, Jan, 2017 
'''

import platform
from numpy import array, floor, nonzero, logical_or, logical_and, logical_not, zeros
from pyproj import Proj, transform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('TkAgg')


class RasterData(object):
    """
    A raster representation of the point set and munipulations on it
    """

    __rasterData = None  # A m-by-n-by-p array in which the raster is stored
    __voidData = -9999  # A number indicating missing data in the raster
    __cellSize = 0.05  # The spacing between grid cells
    __spatialReference = None # spatial reference info

    # transformation to convert pixel to world coordinates
    tx = ty = 0
    sx = sy =1
    __geoTransform = (tx,ty,sx,sy)

    #Lidar data info
    __points = None # PointSet data from which the raster was made from
    __dataOrigin = "" # A string indicating the point acquiring means: 'terrestrial',
                # 'airborne' , 'mobile', 'slam', 'kinect', 'photogrammetry'
    __dataFeature = ""  # A string indicating the type of data stored  (e.g. range, intensity, etc.)
    __measurement_accuracy= 0.15 # The accuracy of a depth/height/range measurement
    __mean_roughness = .02 # roughness estimation of the raster data


    def __init__(cls, data, gridSpacing, **kwargs):
        """
        constructor
        :param points: PointSet object from which the panorama will be created
        :param gridSpacing: The spacing between grid cells

        :param voidData: A number indicating missing data in the raster
        :param geoTrnasform: The transformation needed to convert from pixel to world coordinates (tx, ty, sx, sx)
        :param spatialRefEpsg: The EPSG code of spatial reference of the raster given

        """

        cls.__cellSize = gridSpacing
        cls.__rasterData = data

        cls.setValues(**kwargs)

    @classmethod
    def setValues(cls, **kwargs):
        """
        sets self values
        :param in kwargs: extent, ncolumns, nrows, cellSize, noDataValue,

        """
        params = {'voidData': cls.__voidData,
                  'geoTransform': (cls.__geoTransform),
                  'spatial_reference': cls.__spatialReference,
                  'cellSize': cls.__cellSize,
                  'points': cls.__points,
                  'dataOrigin': cls.__dataOrigin,
                  'dataFeature': cls.__dataFeature,
                  'accuracy': cls.__measurement_accuracy,
                  'roughness': cls.__mean_roughness
                  }
        params.update(kwargs)
        cls.__geoTransform = params['geoTransform']
        cls.__spatialReference = params['spatial_reference']

        cls.__voidData = params['voidData']
        cls.__cellSize = params['cellSize'],
        cls.__points = params['points']
        cls.__dataOrigin = params['dataOrigin']
        cls.__dataFeature = params['dataFeature']
        cls.__measurement_accuracy = params['accuracy']
        cls.__mean_roughness = params['roughness']

    def pixelToWorld(self, row, col):
        """

        :param row:
        :param col:
        :return:
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
        :return: the row and column that correspond to the given point (tuple of two float numbers)
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
        outOfBoundsTest = logical_or(row < 0, logical_or(row >= self.__rasterData.shape[0],
                                                         logical_or(col < 0, col >= self.__rasterData.shape[1])))
        values[nonzero(outOfBoundsTest)[0]] = None

        exactPointsTest = logical_and(abs(row - round(row)) < 1e-8, abs(col - round(col)) < 1e-8)
        exactPointsIdx = nonzero(exactPointsTest)[0]
        if len(exactPointsIdx) > 0:
            values[exactPointsIdx] = self.__rasterData[row[exactPointsIdx], col[exactPointsIdx]]

        otherPoints = nonzero(logical_and(logical_not(outOfBoundsTest),
                                          logical_not(exactPointsTest)))[0]

        if len(otherPoints) > 0:
            if method == 'nearest':
                return self.__rasterData[round(row[otherPoints]), round(col[otherPoints])]
            elif method == 'bilinear':
                try:
                    intRow, intCol = int_(floor(row[otherPoints])), int_(floor(col[otherPoints]))
                    z00 = self.__rasterData[intRow, intCol]
                    z10 = self.__rasterData[intRow, intCol + 1]
                    z01 = self.__rasterData[intRow + 1, intCol]
                    z11 = self.__rasterData[intRow + 1, intCol + 1]

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
            xv = arange(self.__rasterData.shape[1])
            yv = arange(self.__rasterData.shape[0])

        xs, ys = meshgrid(xv, yv)
        xs = xs.reshape((-1, ))
        ys = ys.reshape((-1, ))
        rasterData = self.__rasterData[xs, ys].reshape((-1, ))

        xs = (self.translationX + self.scaleX * xs)
        ys = (self.translationY + self.scaleY * ys)

        if not (spatialRef is None) and spatialRef != self.__spatialReference:
            rasterProj = Proj(init='epsg:' + self.__spatialReference)
            pointProj = Proj(init='epsg:' + spatialRef)
            xs, ys = transform(rasterProj, pointProj, xs, ys)

        return array([xs, ys, rasterData]).T

#------------------ PROPERTIES --------------------------
    @property
    def data(cls):
        """
        :return: return the image data
        """
        return cls.__rasterData

    @property
    def spatialReference(self):
        return self.__spatialReference

    @property
    def translationX(self):
        return self.__geoTransform[0]

    @property
    def translationY(self):
        return self.__geoTransform[1]

    @property
    def scaleX(self):
        return self.__geoTransform[2]

    @property
    def scaleY(self):
        return self.__geoTransform[3]

    @property
    def shape(cls):
        return cls.__rasterData.shape

    @property
    def rows(cls):
        return cls.__rasterData.shape[0]

    @property
    def cols(cls):
        return cls.__rasterData.shape[1]

    @property
    def resolution(cls):
        """
        :return: raster resolution
        """
        return cls.__cellSize

    @property
    def accuracy(cls):
        """
        :return: measurement accuracy
        """
        return cls.__measurement_accuracy

    @property
    def roughness(cls):
        """
        :return: measurement accuracy
        """
        return cls.__mean_roughness



    def min_range(self):
        """
        :return: the closest range in the data
        """


if __name__ == '__main__':
    pass
