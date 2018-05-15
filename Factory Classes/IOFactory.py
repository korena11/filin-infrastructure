# Updated Last on 13/06/2014 14:07

import re
from string import split
from sys import exc_info
from traceback import print_tb

import h5py
import numpy as np
from numpy import array, asarray, hstack, tile, ndarray, where, savetxt
from osgeo import gdal

from BaseData import BaseData
from BaseProperty import BaseProperty
from ColorProperty import ColorProperty
from MyTools import CreateFilename
from PointSet import PointSet
from PointSubSet import PointSubSet
from RasterData import RasterData
from SegmentationProperty import SegmentationProperty
from SphericalCoordinatesProperty import SphericalCoordinatesProperty
from shapefile import Writer, Reader, POINTZ


class IOFactory:
    """
    Class for loading and saving point clouds or rasters from files
    Create PointSet from different types of input files: .pts, .xyz
    
    Write PointSet Data to different types of file: .pts(TODO), .xyz(TODO), shapeFile

    """

    # ---------------------------READ -------------------------------
    @classmethod
    def load(cls, filename, classname, **kwargs):
        """
        Loads an object file (json or hdf5)

        .. warning:: The option for json is not implemented

        :param filename: the path
        :param classname: the property or data class that the filename stores.

        :type filename: str
        :type classname: BaseProperty or BaseData

        :return: the property or dataset

        """

        f, ext = CreateFilename(filename, mode = 'r')
        if ext == 'h5':
            print("Keys: %s" % f.keys())

            params = {}
            data = None
            attribute_value = []
            attribute_name = []
            obj = None

            # Get the data
            for group_name in f.keys():
                attribute_value = []
                attribute_name = []

                items = f[group_name].attrs
                matched = re.match('\_(.*)\_\_(.*)', items.keys()[0])
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

            params2 = dict(zip(attribute_name, attribute_value))
            params.update(params2)

            if issubclass(classname, BaseData):
                data.setValues(**params)
                obj = data
            elif issubclass(classname, BaseProperty):
                obj = classname(data, *params.values())

        f.close()

        return obj

    @classmethod
    def ReadPts(cls, filename, pointsetlist = None, **kwargs):
        """
        Reading points from .pts file
        Creates one or more PointSet objects, returned through pointSetList 

        :param fileName: name of .pts file
        :param pointSetList: place holder for created PointSet objects
        :param merge: True to merge points in file; False do no merge

        :type filename: str
        :type pointsetlist: list
        :type merge: bool

        :return: Number of PointSet objects created
        :rtype: int

        """

        # Opening file and reading all lines from it
        fin = open(filename)
        fileLines = fin.read()
        fin.close()
        numLines = 10

        # Splitting into list of lines 
        lines = split(fileLines, '\n')
        del fileLines

        # Removing the last line if it is empty
        while (True):
            numLines = len(lines)
            if lines[numLines - 1] == "":
                numLines -= 1
                lines = lines[0: numLines]
            else:
                break

        # Removing header line
        readLines = 0
        while readLines < numLines:
            data = []
            # Read number if points in current cloud
            numPoints = int(lines[readLines])
            # Read the points
            currentLines = lines[readLines + 1: readLines + numPoints + 1]
            # Converting lines to 3D Cartesian coordinates Data                
            data = map(cls.__splitPtsString, currentLines)
            # advance lines counter
            readLines += numPoints + 1

            data = array(data)

            xyz = asarray(data[:, 0:3])
            rgb = None
            intensity = None

            if numPoints == 1:
                kwargs['vertList'].append(xyz)
            else:
                if data.shape[1] == 6:
                    rgb = np.asarray(data[:, 3:6], dtype = np.uint8)
                if data.shape[1] == 7:
                    rgb = np.asarray(data[:, 4:7], dtype = np.uint8)
                if data.shape[1] == 4 or data.shape[1] == 7:
                    intensity = np.asarray(data[:, 3], dtype = np.int)
                    # Create the PointSet object
                pointSet = PointSet(xyz, rgb = rgb, intensity = intensity)
                pointSet.setPath(filename)
                pointsetlist.append(pointSet)

        del lines

        if 'merge' in kwargs:

            if kwargs.get('merge', True):
                return cls.__mergePntList(pointSetList)
            else:
                return len(pointsetlist)
        else:
            return len(pointsetlist)

    @classmethod
    def ReadPtx(cls, fileName, pointSetList):
        """
        Reading points from .ptx file
        Creates one or more PointSet objects, returned through pointSetList 
        
        :param fileName: name of .ptx file
        :param pointSetList: place holder for created PointSet objects

        :type fileName: str
        :type pointSetList: list
            
        :return: Number of PointSet objects created
        :rtype: int
 
        """
        pointSet = None
        # Opening file and reading all lines from it
        fin = open(fileName)
        fileLines = fin.read()
        fin.close()

        # Splitting into list of lines 
        lines = split(fileLines, '\n')
        del fileLines

        # Removing the last line if it is empty
        while (True):
            numLines = len(lines)
            if lines[numLines - 1] == "":
                numLines -= 1
                lines = lines[0: numLines]
            else:
                break

        # Removing header line
        data = []
        #         currentLines = lines[10::]
        # Converting lines to 3D Cartesian coordinates Data     
        linesLen = map(lambda x: len(x), lines)
        line2del = (where(np.asarray(linesLen) < 5)[0])

        if len(line2del) > 1 and line2del[0] - line2del[1] == -1:
            line2del = line2del[-2::-2]  # there two lines one after another with length 1, we need the first one
        for i2del in line2del:
            del lines[i2del:i2del + 10]
        data = map(cls.__splitPtsString, lines)
        line2del = where(asarray(data)[:, 0:4] == [0, 0, 0, 0.5])[0]
        data = np.delete(data, line2del, 0)

        data = array(data)

        xyz = asarray(data[:, 0:3])
        if data.shape[1] == 6:
            rgb = np.asarray(data[:, 3:6], dtype = np.uint8)
            pointSet = PointSet(xyz, rgb = rgb)
        if data.shape[1] == 7:
            rgb = np.asarray(data[:, 4:7], dtype = np.uint8)
            intensity = np.asarray(data[:, 3], dtype = np.int)
            pointSet = PointSet(xyz, rgb = rgb, intensity = intensity)
        if data.shape[1] == 4 or data.shape[1] == 7:
            intensity = np.asarray(data[:, 3], dtype = np.int)
            pointSet = PointSet(xyz, intensity = intensity)

        pointSet.setPath(fileName)
        # Create the List of PointSet object            
        pointSetList.append(pointSet)

        del lines

        return len(pointSetList)

    @classmethod
    def ReadXYZ(cls, fileName, pointSetList):
        """
        Reading points from .xyz file
        Creates one PointSet objects returned through pointSetList
        

        :param fileName: name of .xyz file
        :param pointSetList: place holder for created PointSet object

        :type fileName: str
        :type pointSetList: list
            
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
        pointSetList.append(pointSet)

        return len(pointSetList)

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
            polylinePoints = map(cls.__ConvertRecodrsToPoints, shapeRecord)

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
            print "Unexpected error: ", exc_info()[0]
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
            print "Unexpected error: ", exc_info()[0]
            print_tb(exc_info()[2])
            return None

        ncols = np.int(filelines[0].split(' ')[-1])
        nrows = np.int(filelines[1].split(' ')[-1])
        xllcorner = np.float32(filelines[2].split(' ')[-1])
        yllcorner = np.float32(filelines[3].split(' ')[-1])
        cellsize = np.float32(filelines[4].split(' ')[-1])
        nodata_value = np.float32(filelines[5].split(' ')[-1])

        tmp = lambda x: np.float32(x.split(' ')[:-1])

        data = array(map(tmp, filelines[6:]))
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
    def WriteToShapeFile(cls, pointSet, fileName, **kwargs):
        """
        Exporting points to shapefile
        
        :param pointSet: A PointSet\PointSubSet object with the points to be extracted
        :param fileName: Full path and name of the shapefile to be created (not including the extension)
        :param kwargs: Additional properties can be sent using kwargs which will be added as attributes in the shapfile

        :type pointSet: PointSet or PointSubSet
        :type fileName: str

        :return:


        """
        if (pointSet.Z != None):
            fieldList = ['X', 'Y', 'Z']
        else:
            fieldList = ['X', 'Y']

        attributes = pointSet.ToNumpy

        if (pointSet.Intensity != None):
            fieldList.append('intensity')
            attributes = hstack([attributes, pointSet.Intensity.reshape((pointSet.Size, 1))])
        if (pointSet.RGB != None):
            fieldList.append('r')
            fieldList.append('g')
            fieldList.append('b')
            attributes = hstack([attributes, pointSet.RGB])

        for auxPropertyName, auxProperty in kwargs.iteritems():
            if (isinstance(auxProperty, ColorProperty)):
                fieldList.append(auxPropertyName + '_r')
                fieldList.append(auxPropertyName + '_g')
                fieldList.append(auxPropertyName + '_b')
                attributes = hstack([attributes, pointSet.RGB])
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

        map(w.field, fieldList, tile('F', len(fieldList)))
        if (pointSet.Z != None):
            map(w.point, pointSet.X, pointSet.Y, pointSet.Z)

        else:
            map(w.point, pointSet.X, pointSet.Y)

        # map(w.record, attributes2)
        w.records = map(ndarray.tolist, attributes)

        w.save(fileName)

    # ---------------------------PRIVATES -------------------------------
    @classmethod
    def __ConvertRecodrsToPoints(cls, shapeRecord):
        """ 
        Converting polyline into points
        
        :param fileName: Full path and name of the shapefile to be created (not including the extension)

        """

        points = array(shapeRecord.shape.points)
        return points

    @classmethod
    def __mergePntList(cls, pointSetList):
        '''
        Merging several pointset

        :param pointSetList: a list of PointSets

        :return refPntSet: merged PointSet

        '''
        # TODO: changed from MegrePntList to __MergePntList. CHECK WHERE WAS IN USE!

        list_length = len(pointSetList)
        if list_length > 1:
            refPntSet = pointSetList[0]
            fields_num = refPntSet.FieldsDimension
            refPntSet.setPath(pointSetList[0].path)
            for pntSet in pointSetList[1::]:
                if fields_num == 7:
                    refPntSet.AddData2Fields(pntSet.ToNumpy(), field = 'XYZ')
                    refPntSet.AddData2Fields(pntSet.RGB, field = 'RGB')
                    refPntSet.AddData2Fields(pntSet.Intensity, field = 'Intensity')
                elif fields_num == 6:
                    refPntSet.AddData2Fields(pntSet.ToNumpy(), field = 'XYZ')
                    refPntSet.AddData2Fields(pntSet.RGB, field = 'RGB')
                elif fields_num == 4:
                    refPntSet.AddData2Fields(pntSet.ToNumpy(), field = 'XYZ')
                    refPntSet.AddData2Fields(pntSet.Intensity, field = 'Intensity')
                else:
                    refPntSet.AddData2Fields(pntSet.ToNumpy(), field = 'XYZ')

            return refPntSet
        else:
            return pointSetList[0]

    @classmethod
    def __splitPtsString(cls, line):
        """
        Extracting from a string the Data of a point (3D coordinates, intensity, color)

        :param line: A string containing a line from a .pts file

        :return: all existing Data of a point

        :rtype: ndarray

        """
        tmp = split(line, ' ')
        return map(float, tmp)

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

        for attr in items.keys():
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
                                spatialReference = items[u'_RasterData__' + 'spatialReference']
                            except:
                                spatialReference = None
                            dataset = cls.rasterFromAscFile(dataset, spatialReference)

            else:
                if attr_dataclass is not 'BaseData':
                    classname = attr_dataclass

                attribute_name.append(matched.group(2))
                attribute_value.append(items[attr])

        params = dict(zip(attribute_name, attribute_value))

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
    # fileName = '..\Sample Data\Geranium2Clouds.pts'
    # IOFactory.ReadPts(fileName, pointSetList)
    # ===========================================================================

    # =================shapefile test==================================
    polylineFileName = r'E:\My Documents\Projects\IOLR\HaBonim\polyRimSample.shp'
    IOFactory.ReadShapeFile(polylineFileName, pointSetList)
