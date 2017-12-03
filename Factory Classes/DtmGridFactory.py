from osgeo import gdal, osr
from DtmGrid import DtmGrid
from sys import exc_info
from traceback import print_tb
from numpy import array, float32


class DtmGridFactory:

    @classmethod
    def DtmGridFromGDAL(cls, path):
        try:
            ds = gdal.Open(path)
            return DtmGrid(ds.ReadAsArray(), ds.GetGeoTransform(), ds.GetProjection().split('\"')[-2])
        except:
            print "Unexpected error: ", exc_info()[0]
            print_tb(exc_info()[2])
            return None

    @classmethod
    def DtmGridFromAscFile(cls, path, projection=None):
        try:
            fin = open(path, 'r')
            filelines = fin.readlines()
            fin.close()
        except:
            print "Unexpected error: ", exc_info()[0]
            print_tb(exc_info()[2])
            return None

        ncols = float32(filelines[0].split(' ')[-1])
        nrows = float32(filelines[1].split(' ')[-1])
        xllcorner = float32(filelines[2].split(' ')[-1])
        yllcorner = float32(filelines[3].split(' ')[-1])
        cellsize = float32(filelines[4].split(' ')[-1])

        tmp = lambda x: float32(x.split(' ')[:-1])

        data = array(map(tmp, filelines[6:]))
        return DtmGrid(data, (xllcorner, cellsize, 0.0, yllcorner, 0.0, cellsize), projection)

    @classmethod
    def ResampleDtmGrid(cls, dtmGrid, shiftRow=0, shiftCol=0, resampleRate=1):
        from numpy import arange, meshgrid

        minRow = 0 if shiftRow < 0 else shiftRow
        maxRow = dtmGrid.shape[0] - 1 if shiftRow > 0 else dtmGrid.shape[0] - shiftRow

        minCol = 0 if shiftCol < 0 else shiftCol
        maxCol = dtmGrid.shape[1] - 1 if shiftCol > 0 else dtmGrid.shape[1] - shiftCol

        rows = arange(minRow, maxRow, resampleRate)
        cols = arange(minCol, maxCol, resampleRate)

        nRows = rows.shape[0]
        nCols = cols.shape[0]

        try:
            rows, cols = meshgrid(rows, cols)
            rows = rows.reshape((-1, ))
            cols = cols.reshape((-1, ))

            data = dtmGrid.intepolateAtPixel(rows, cols)
            data = data.reshape((nRows, nCols))
        except:
            print "Unexpected error: ", exc_info()[0]
            print_tb(exc_info()[2])
            return None

        return DtmGrid(data, (dtmGrid.translationX + shiftCol * dtmGrid.scaleX, dtmGrid.scaleX * resampleRate, 0.0,
                              dtmGrid.translationY + shiftRow * dtmGrid.scaleY, 0.0, dtmGrid.scaleY * resampleRate),
                       dtmGrid.spatialReference)

if __name__ == '__main__':
    DtmGridFactory.DtmGridFromAscFile('../Data/MMG DTM/dem_1_50.asc')