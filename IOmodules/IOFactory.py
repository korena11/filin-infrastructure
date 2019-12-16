# Updated Last on 13/06/2014 14:07

import h5py
import re
import warnings
from numpy import array
from sys import exc_info
from traceback import print_tb

import ReadFunctions
import SaveFunctions
from DataClasses.BaseData import BaseData
from DataClasses.PointSet import PointSet
from DataClasses.PointSubSet import PointSubSet
from DataClasses.RasterData import RasterData
from IO_Tools import CreateFilename
from Properties.BaseProperty import BaseProperty
from Properties.Color.ColorProperty import ColorProperty


# from osgeo import gdal
# from shapefile import Reader


class IOFactory:
    """
    Class for loading and saving point clouds or rasters from files

    Creates PointSet from: .pts, .xyz

    Creates RasterData from: .asc, .txt
    
    Write PointSet Data to different types of file: .pts(TODO), .xyz(TODO), shapeFile

    """

    # ---------------------------READ -------------------------------
    @classmethod
    def load(cls, filename, classname=None, pointsetlist=None, colorlist=None, merge=True, **kwargs):
        """
        Loads an object file (json or hdf5)

        .. warning:: The option is not implemented for json

        :param filename: the path
        :param classname: the property or data class that the filename stores
        :param pointsetlist: a list to store pointset list if more than one cloud is in the file (applicaple for pts, ptx and shapefiles)
        :param colorlist: a list to store colors proeprty, if colors are within the file (applicaple for pts and ptx)
        :param merge: flag to merge multiple clouds into one pointset (applicaple for pts)

        :type filename: str
        :type classname: type
        :type pointsetlist: list
        :type colorlist: list
        :type merge: bool

        :return: the property or dataset

        """

        f, ext = CreateFilename(filename, mode = 'r')
        if ext == 'h5':
            try:
                obj = cls.ReadHDF(f, classname, **kwargs)
            except:
                warnings.warn('No class type assigned for hdf file')
        elif ext == 'p' or ext == 'pickle' or ext == 'pkl':
            obj = cls.ReadPickle(f, classname)

        elif ext == 'pts':
            obj = cls.ReadPts(filename, pointsetlist=pointsetlist, colorslist=colorlist, merge=merge)

        elif ext == 'xyz':
            obj = cls.ReadXYZ(filename)

        elif ext == 'ptx':
            obj = cls.ReadPtx(filename, pointsetlist=pointsetlist, colorslist=colorlist)
        elif ext == 'shp':
            obj = cls.ReadShapeFile(filename, pointsetlist)

        f.close()

        return obj

    @classmethod
    def ReadPickle(cls, fileobj, classname):
        """
        Reads any pickle and assigns it according to given class.

        :param fileobj: the filename
        :param classname: the class to assign the pickle to

        :type fileobj: file
        :type classname: type

        :return: the object loaded from the pickle

        :rtype: BaseData or BaseProperty
        """
        import _pickle
        attrs = _pickle.load(fileobj)
        attrs_new = {}

        if 'current' in attrs:
            attrs.pop('current')

        for key in attrs:
            matched = re.match('\_(.*)\_\_(.*)', key)
            if matched is None:
                attrs_new.update({key: attrs[key]})
            else:
                attrs_new.update({matched.group(2): attrs[key]})

        if 'data' in attrs_new:
            data = attrs_new.pop('data')
        else:
            try:
                filename = fileobj.name.split('.')
                datafilename = filename[0] + '__data' + '.' + filename[1]
                if 'indices' in attrs_new:
                    data = cls.load(datafilename, PointSubSet)
                else:
                    data = cls.load(datafilename, PointSet)
            except:
                datafilename = attrs_new['path']
                data = cls.load(datafilename, PointSet)

                if 'indices' in attrs_new:
                    data = PointSubSet(data, attrs_new['indices'])

        if issubclass(classname, BaseData):
            obj = classname(data, **attrs_new)

        if issubclass(classname, BaseProperty):
            # TODO: if the data is empty, should load from the path or look for a xx_data file

            obj = classname(data)
            obj.load(**attrs_new)

        return obj

    @classmethod
    def ReadHDF(cls, f, classname, **kwargs):

        print(("Keys: %s" % list(f.keys())))

        params = {}
        data = None
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
    def ReadPts(cls, filename, pointsetlist=None, colorslist=None, merge=True):
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
    def ReadPtx(cls, filename, pointsetlist=None, colorslist=None, trasformationMatrices=None,
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
        :type colorslist: list
        :type trasnformationMatrices: list

        :return: pointSet list

        :rtype: list

        .. warning:: Doesn't read the transformation matrices.

        """
        return ReadFunctions.ReadPtx(filename, pointsetlist, colorslist, trasformationMatrices, remove_empty)

    @classmethod
    def ReadXYZ(cls, fileName, pointsetlist=None, merge=True):
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
        return ReadFunctions.ReadXYZ(fileName, pointsetlist, merge)

    @classmethod
    def ReadLAS(cls, fileName, classification_flag=False):
        """
        Reads LAS or LAZ file. Can return classification property in addition to the point cloud

        :param fileName: path to file (LAS or LAZ)
        :param classification_flag: a flag whether to return a classification property or not (default: False)

        :type fileName: str
        :type classification_flag: bool

        :return: pointcloud, classification

        :rtype: PointSet, ClassificationProperty
        """

        return ReadFunctions.ReadLAS(fileName, classification_flag)

    @classmethod
    def ReadPly(cls, filename, returnAdditionalAttributes=True):
        """
        Reading ply file
        The method returns a PointSet object that contains the 3-D coordinates of all vertices in the ply file and
        their intensity values. If additional attributes exist they are returned as a dictionary with the attribute
        names as the keys

        :param filename: path to *.ply file
        :param returnAdditionalAttributes: Indicator whether or not return the additional attributes that exist

        :type filename: str
        :type returnAdditionalAttributes: bool

        :return: PointSet object and dictionary with additional properties (optional)

        :rtype: tuple of PointSet object and a dictionary
        """
        return ReadFunctions.ReadPly(filename, returnAdditionalAttributes)


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
                pointSet.path = fileName
                pointSetList.append(pointSet)
        else:
            return 0

    @classmethod
    def read2_PointSetOpen3D(cls, file_path, voxel_size=-1, print_bb=False):
        """
        Reads a file into a PointSetOpen3D object

        :param file_path: Path of pointcloud file
        :param voxel_size: If >0 then decimate point cloud with this parameter as min points distance
        :param print_bb: Print boundarybox values

        :type: str
        :type: Positive double
        :type: bool

        :return: pointsetExtra Object
        :rtype: PointSetExtra

        """
        return ReadFunctions.read2_PointSetOpen3D(file_path, voxel_size=-1, print_bb=False)

    @classmethod
    def GetCurvatureFilePath(cls, folderPath, dataName, currentFileIndex, localNeighborhoodParameters, decimationRadius,
                             testRun):
        """
        Gets the curvature files according to path

        :param folderPath: Path to curvature-files folder
        :param dataName: name of the new file
        :param currentFileIndex: indexing for saving (file1.txt, file2.txt, etc)
        :param localNeighborhoodParameters: dictionary holding the neighborhood parameters (radius and maximum number of neighbors)
        :param decimationRadius: minimal distance between two points for downsampling .
        :param testRun: If true then only temporary files will be saved that are cleaned at each run.

        :TODO: Elia -- please provide better explanation

        :type folderPath: str
        :type dataName: str
        :type currentFileIndex: int
        :type localNeighborhoodParameters: dict
        :type decimationRadius: float
        :type testRun: bool

        :return: full filename and path for loading (or saving) curvature files

        :rtype: str

        """
        return ReadFunctions.GetCurvatureFilePath(folderPath, dataName, currentFileIndex, localNeighborhoodParameters,
                                                  decimationRadius, testRun)
    @classmethod
    def rasterFromGDAL(cls, path, cellsize):
        """
        Read raster files (tif, GeoTIf, png etc.) and load to RasterData

        :param path: path to raster
        :param cellsize: raster cell size

        :type path: str
        :type cellsize: float

        :return: raster data

        :rtype: RasterData
        """
        import gdal

        try:
            ds = gdal.Open(path)
            data = ds.ReadAsArray()

        except IOError:
            print("Unexpected error: ", exc_info()[0])
            print_tb(exc_info()[2])
            return None

        geoTransform = ds.GetGeoTransform()

        try:
            projection = ds.GetProjection().split('\"')[-2]
        except:
            projection = 'CS_local'

        return RasterData(data, gridSpacing=cellsize,
                          geoTransform=(geoTransform[0], geoTransform[3], geoTransform[1], geoTransform[-1]),
                          spatial_reference=projection, path=path)


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
        return ReadFunctions.rasterFromAscFile(path, projection)

    # ---------------------------WRITE -------------------------------
    @classmethod
    def saveDataset(cls, dataset, filename, name='dataset'):
        """
        Save dataset according to filename extensioin.

        The extension depends on the filename extension.

        :param dataset: the dataset to save
        :param filename: the filename to save the dataset to
        :param name: for h5 files, the name of the group

        :type dataset: BaseData
        :type filename: str
        :type name: str

        :return:
        """
        if isinstance(filename, str):
            filename, extension = CreateFilename(filename)

        if isinstance(filename, h5py.File):
            SaveFunctions.save_dataset_h5(dataset, filename, name, True)

        else:
            try:
                SaveFunctions.pickleDataset(dataset, filename)
            except:
                from warnings import warn

                warn(IOError, 'Not sure how to save')

        return filename

    @classmethod
    def saveProperty(cls, property_class, filename, save_dataset=False, **kwargs):
        """
        Saves a property to hdf5 or json file

        Default: hdf5

        .. warning:: Json file save is unavailable

        :param file: path or file object
        :param property_class: the property which is needed to be saved
        :param save_dataset: flag whether to save the dataset that the property relates to or not. Default: True

        :type property_class: BaseProperty or subclasses
        :type filename: str
        :type save_dataset: bool

        :return:

        """
        import SaveFunctions
        import h5py

        if isinstance(filename, str):
            filename, extension = CreateFilename(filename)

        if isinstance(filename, h5py.File):
            SaveFunctions.save_property_h5(property_class, filename, save_dataset)

        else:
            try:
                SaveFunctions.pickleProperty(property_class, filename, save_dataset)
            except IOError:
                from warnings import warn
                warn(IOError, 'Not sure how to save')

    @classmethod
    def WriteToPts(cls, points, path):
        '''
        Write to pts file

            :param points: PointSet
            :param path: to the directory of a new file + file name
        '''

        SaveFunctions.WriteToPts(points, path)

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
        SaveFunctions.WriteToShapeFile(pointSet, fileName, colors, **kwargs)



    @staticmethod
    def curve2shp(curves, filename):
        """
        TODO: make function
        Turn curves to shapefiles

        :param curves: curves as represented by matplotlib.plot
        :param filename: file to save the curve

        .. warning::
          Empty function.

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

        .. warning:: Doesn't work for data classes other than PointSet or RasterData

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
