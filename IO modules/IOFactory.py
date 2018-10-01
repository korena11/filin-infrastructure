# Updated Last on 13/06/2014 14:07

import re
import warnings
from sys import exc_info
from traceback import print_tb

import h5py
import numpy as np
from numpy import array, hstack, tile, ndarray, savetxt
from osgeo import gdal

import ReadFunctions
from BaseData import BaseData
from BaseProperty import BaseProperty
from ColorProperty import ColorProperty
from IO_Tools import CreateFilename
from PointSet import PointSet
from PointSubSet import PointSubSet
from RasterData import RasterData
from SegmentationProperty import SegmentationProperty
from SphericalCoordinatesProperty import SphericalCoordinatesProperty
from shapefile import Writer, Reader, POINTZ


class IOFactory:
    """
    Class for loading and saving point clouds or rasters from files

    Creates PointSet from: .pts, .xyz

    Creates RasterData from: .asc, .txt
    
    Write PointSet Data to different types of file: .pts(TODO), .xyz(TODO), shapeFile

    """

    # ---------------------------READ -------------------------------
    @classmethod
    def load(cls, filename, classname, **kwargs):
        """
        Loads an object file (json or hdf5)

        .. warning:: The option is not implemented for json

        :param filename: the path
        :param classname: the property or data class that the filename stores -

        :type filename: str
        :type classname: BaseProperty or BaseData

        :return: the property or dataset

        """

        f, ext = CreateFilename(filename, mode = 'r')
        if ext == 'h5':
            try:
                obj = cls.ReadHDF(f, classname, **kwargs)
            except:
                warnings.warn('No class type assigned for hdf file')

        f.close()

        return obj

    @classmethod
    def ReadHDF(cls, f, classname, **kwargs):

        print(("Keys: %s" % list(f.keys())))

        params = {}
        data = None
        attribute_value = []
        attribute_name = []
        obj = None

        # Get the data
        for group_name in list(f.keys()):
            attribute_value = []
            attribute_name = []

            items = f[group_name].attrs
            matched = re.match('\_(.*)\_\_(.*)', list(items.keys())[0])
            loaded_classname = matched.group(1)

            group = f[group_name]
            obj = cls.__loadata_hdf5(group, loaded_classname)

            if len(group_name.split('dataset')) > 1:
                data = obj

                continue

            if isinstance(obj, dict):
                params.update(obj)
            else:
                attribute_name.append(group_name)
                attribute_value.append(obj)

        params2 = dict(list(zip(attribute_name, attribute_value)))
        params.update(params2)

        if issubclass(classname, BaseData):
            data.setValues(**params)
            obj = data
        elif issubclass(classname, BaseProperty):
            obj = classname(data, *list(params.values()))

        return obj

    @classmethod
    def ReadPts(cls, filename, pointsetlist=list(), colorslist=list(), merge=True):
        """
        Reading points from .pts file. If the pts file holds more than one PointSet merge into one PointSet (unless told
        otherwise).

        :param fileName: name of .pts file

        **Optionals**

        :param pointsetlist: placeholder for created PointSet objects
        :param colorslist: placeholder for ColorProperty for PointSet object(s)
        :param merge: merge points in file into one PointSet or not. Default: True.

        :type filename: str
        :type pointsetlist: list
        :type colorslist: list
        :type merge: bool

        :return: The created PointSet or the list of the PointSets created and the ColorProperty that belongs to it

        :rtype: PointSet

        """
        return ReadFunctions.ReadPts(filename, pointsetlist, colorslist, merge)


    @classmethod
    def ReadPtx(cls, filename, pointsetlist=list(), colorslist=list(), trasformationMatrices=list(),
                remove_empty=True):

        """
        Reads .ptx file, created by Leica Cyclone

        File is built according to:
        https://w3.leica-geosystems.com/kb/?guid=5532D590-114C-43CD-A55F-FE79E5937CB2

        :param filename: path to file + file

        *Optionals*

        :param pointsetlist: list that holds all the uploaded PointSet
        :param colorslist: list that holds all the color properties that relate to the PointSet
        :param transformationMatrices: list that holds all the transformation properties that relate to the PointSet

        :type filename: str
        :type pointsetlist: list
        :type colorslist: list of ColorProperty.ColorProperty
        :type trasnformationMatrices: list of TransformationMatrixProperty

        :return: pointSet list

        :rtype: list

        .. warning:: Doesn't read the transformation matrices.

        """
        return ReadFunctions.ReadPtx(filename, pointsetlist, colorslist, trasformationMatrices, remove_empty)


    @classmethod
    def ReadXYZ(cls, fileName, pointsetlist=list(), merge=True):
        """
        Reading points from .xyz file
        Creates one PointSet objects returned through pointSetList
        

        :param fileName: name of .xyz file
        :param pointsetlist: place holder for created PointSet object

        :type fileName: str
        :type pointsetlist: list
            
        :return: Number of PointSet objects created

        :rtype: int
 
        """
        parametersTypes = np.dtype({'names': ['name', 'x', 'y', 'z']
                                       , 'formats': ['int', 'float', 'float', 'float']})

        #         parametersTypes = np.dtype({'names':['x', 'y', 'z']
        #                                , 'formats':['float', 'float', 'float']})

        imported_array = np.genfromtxt(fileName, dtype = parametersTypes, filling_values = (0, 0, 0, 0))

        xyz = imported_array[['x', 'y', 'z']].view(float).reshape(len(imported_array), -1)

        pointSet = PointSet(xyz)
        pointSet.setPath(fileName)
        pointsetlist.append(pointSet)

        if merge:
            pointsetlist = np.array(pointsetlist)
        return len(pointsetlist)

    @classmethod
    def ReadShapeFile(cls, fileName, pointSetList):
        # TODO: make small functions from other kinds of shapefiles rather than polylines
        """
         Importing points from shapefiles

         :param fileName:  Full path and name of the shapefile to be created (not including the extension)
         :param pointSetList: place holder for created PointSet objects

         :type fileName: str
         :type pointSetList: list

         """
        shape = Reader(fileName)

        if shape.shapeType == 3:
            shapeRecord = shape.shapeRecords()
            polylinePoints = list(map(cls.__ConvertRecodrsToPoints, shapeRecord))

            for i in (polylinePoints):
                pointSet = PointSet(i)
                pointSet.setPath(fileName)
                pointSetList.append(pointSet)
        else:
            return 0

    @classmethod
    def rasterFromGDAL(cls, path):
        try:
            ds = gdal.Open(path)

            return RasterData(ds.ReadAsArray(), ds.GetGeoTransform(),
                              spatial_reference = ds.GetProjection().split('\"')[-2])
        except:
            print("Unexpected error: ", exc_info()[0])
            print_tb(exc_info()[2])
            return None

    @classmethod
    def rasterFromAscFile(cls, path, projection = None):
        """
        Reads raster from .txt or .asc files

        :param path: path+filename
        :param projection:

        :type path: str

        :return: a RasterData object

         :rtype: RasterData
        """
        try:
            fin = open(path, 'r')
            filelines = fin.readlines()
            fin.close()
        except:
            print("Unexpected error: ", exc_info()[0])
            print_tb(exc_info()[2])
            return None

        ncols = np.int(filelines[0].split(' ')[-1])
        nrows = np.int(filelines[1].split(' ')[-1])
        xllcorner = np.float32(filelines[2].split(' ')[-1])
        yllcorner = np.float32(filelines[3].split(' ')[-1])
        cellsize = np.float32(filelines[4].split(' ')[-1])
        nodata_value = np.float32(filelines[5].split(' ')[-1])

        tmp = lambda x: np.float32(x.split(' ')[:-1])

        data = array(list(map(tmp, filelines[6:])))
        return RasterData(data, gridSpacing = cellsize, geoTransform = (xllcorner, yllcorner, 1., 1.),
                          spatial_reference = projection, voidData = nodata_value, path = path)

    # ---------------------------WRITE -------------------------------
    @classmethod
    def saveProperty(cls, file, property, **kwargs):
        """
        Saves a property to hdf5 or json file

        Default: hdf5

        .. warning:: Json file save is unavailable

        :param file: path or file object
        :param property: the property which is needed to be saved
        :param save_dataset: flag whether to save the dataset that the property relates to or not. Default: True

        :type property: BaseProperty or subclasses
        :type filename: str
        :type save_dataset: bool

        :return:

        """
        try:
            property.save(file, **kwargs)
        except:
            print ('Unable to save the file')

    @classmethod
    def WriteToPts(cls, points, path):
        '''
        Write to pts file

            :param points: PointSet
            :param path: to the directory of a new file + file name
        '''

        fields_num = points.FieldsDimension
        if fields_num == 7:
            data = hstack((points.ToNumpy(), points.Intensity, points.RGB))
            fmt = ['%.3f', '%.3f', '%.3f', '%d', '%d', '%d', '%d']
        elif fields_num == 6:
            data = hstack((points.ToNumpy(), points.RGB))
            fmt = ['%.3f', '%.3f', '%.3f', '%d', '%d', '%d']
        elif fields_num == 4:
            data = hstack((points.ToNumpy(), points.Intensity))
            fmt = ['%.3f', '%.3f', '%.3f', '%d']
        else:
            data = points.ToNumpy()
            fmt = ['%.3f', '%.3f', '%.3f']

        savetxt(path, points.Size, fmt = '%long')
        with open(path, 'a') as f_handle:
            savetxt(f_handle, data, fmt, delimiter = '\t', newline = '\n')

    @classmethod
    def WriteToShapeFile(cls, pointSet, fileName, colors=None, **kwargs):
        """
        Exporting points to shapefile
        
        :param pointSet: A PointSet\PointSubSet object with the points to be extracted
        :param fileName: Full path and name of the shapefile to be created (not including the extension)
        :param kwargs: Additional properties can be sent using kwargs which will be added as attributes in the shapfile
        :param colors: ColorProperty object with the points colors

        :type pointSet: PointSet or PointSubSet
        :type fileName: str
        :type colors: ColorProperty


        :return:

        """
        if (pointSet.Z != None):
            fieldList = ['X', 'Y', 'Z']
        else:
            fieldList = ['X', 'Y']

        attributes = pointSet.ToNumpy

        if pointSet.Intensity != None:
            fieldList.append('intensity')
            attributes = hstack([attributes, pointSet.Intensity.reshape((pointSet.Size, 1))])
        if (colors != None):
            fieldList.append('r')
            fieldList.append('g')
            fieldList.append('b')
            attributes = hstack([attributes, colors.RGB])

        for auxPropertyName, auxProperty in kwargs.items():
            if (isinstance(auxProperty, ColorProperty)):
                fieldList.append(auxPropertyName + '_r')
                fieldList.append(auxPropertyName + '_g')
                fieldList.append(auxPropertyName + '_b')
                attributes = hstack([attributes, colors.RGB])

            elif (isinstance(auxProperty, SegmentationProperty)):
                fieldList.append('labels_' + auxPropertyName)
                attributes = hstack([attributes,
                                     auxProperty.GetAllSegments.reshape((pointSet.Size, 1))])

            elif (isinstance(auxProperty, SphericalCoordinatesProperty)):
                fieldList.append('azimuth')
                fieldList.append('elevationAngle')
                fieldList.append('Range')
                attributes = hstack([attributes, auxProperty.ToNumpy])

        w = Writer(POINTZ)

        list(map(w.field, fieldList, tile('F', len(fieldList))))
        if (pointSet.Z != None):
            list(map(w.point, pointSet.X, pointSet.Y, pointSet.Z))

        else:
            list(map(w.point, pointSet.X, pointSet.Y))

        # map(w.record, attributes2)
        w.records = list(map(ndarray.tolist, attributes))

        w.save(fileName)

    @staticmethod
    def curve2shp(curves, filename):
        """
        Turn curves to shapefiles

        :param curves: curves as represented by matplotlib.plot
        :param filename: file to save the curve
        :return:
        """

        # w = Writer(POLYGON)
        #
        # w.field('Area', 'F')
        # w.field('Circumference', 'F')
        #
        # for i in curves:
        #
        #
        # w.poly(curves)
        # w.records = str(range(len(curves)))
        # w.save(filename)


    # ---------------------------PRIVATES -------------------------------
    @classmethod
    def __ConvertRecodrsToPoints(cls, shapeRecord):
        """ 
        Converting polyline into points
        
        :param fileName: Full path and name of the shapefile to be created (not including the extension)

        """

        points = array(shapeRecord.shape.points)
        return points

    # return [float(x) for x in split(line, ' ')]

    @classmethod
    def __loadata_hdf5(cls, group, classname, **kwargs):
        """
         Loads a data file (PointSet or RasterData) from an hdf5 group

        .. warning:: Doesn't work for PointSubSet at the moment

        :param group: a group
        :param classname: the property or data class that the filename stores.

        :type group: h5py.Group
        :type classname: BaseData object

        :return: the dataset object

        """
        # TODO: Add for PointSubSet
        import inspect

        assert isinstance(group, h5py.Group)
        items = group.attrs
        params = {}
        attribute_name = []
        attribute_value = []
        dataset = None

        for attr in list(items.keys()):
            matched = re.match('\_(.*)\_\_(.*)', attr)
            attr_dataclass = matched.group(1)

            if matched.group(2) == 'data':
                dataset = items[attr]

                if isinstance(dataset, str):
                    matched2 = re.match('(.*)\.([a-z].*)', dataset)
                    extension = matched2.group(2)

                    if classname == 'PointSet' or 'BaseData':
                        pointSetList = []
                        if extension == 'pts':
                            dataset = cls.ReadPts(dataset, merge = True)
                        if extension == 'ptx':
                            dataset = cls.ReadPtx(dataset, pointSetList)
                        if extension == 'xyz':
                            dataset = cls.ReadXYZ(dataset, pointSetList)
                        if extension == 'shp':
                            dataset = cls.ReadShapeFile(dataset, pointSetList)

                    elif classname == 'RasterData' or 'BaseData':
                        if extension == 'asc' or 'txt':
                            try:
                                spatialReference = items['_RasterData__' + 'spatialReference']
                            except:
                                spatialReference = None
                            dataset = cls.rasterFromAscFile(dataset, spatialReference)

            else:
                if attr_dataclass is not 'BaseData':
                    classname = attr_dataclass

                attribute_name.append(matched.group(2))
                attribute_value.append(items[attr])

        params = dict(list(zip(attribute_name, attribute_value)))

        if inspect.isclass(dataset):
            if issubclass(dataset, BaseData):
                dataset.setValues(**params)
                return dataset
        else:
            cellSize = params.pop('__cellSize', False)

            if classname == 'RasterData':
                return RasterData(dataset, cellSize, **params)
            elif classname == 'PointSet':
                return PointSet(dataset, **params)

            else:
                return params


if __name__ == "__main__":
    pointSetList = []
    # ====================Point cloud test=================================
    fileName = r'D:\OwnCloud\Data\PCLs\gesher_wall.pts'
    IOFactory.ReadPts(fileName, pointSetList)
    # ===========================================================================

    # =================shapefile test==================================
    # polylineFileName = r'E:\My Documents\Projects\IOLR\HaBonim\polyRimSample.shp'
    # IOFactory.ReadShapeFile(polylineFileName, pointSetList)
