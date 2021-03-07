"""
Reading Functions
=================

Specific functions for file reading. Called from the IOFactory

"""
import sys

if sys.version_info[0] < 3:
    import cPickle as _pickle
else:
    import _pickle
import warnings

# from laspy.file import file
import numpy as np

from Properties.Color.ColorFactory import ColorFactory
from DataClasses.PointSet import PointSet
from DataClasses.RasterData import RasterData
from Properties.Transformations.TransformationMatrixProperty import TransformationMatrixProperty

try:
    from plyfile import PlyData
except:
    warnings.warn('Failed to import plyfile lib. Reading *.ply files will fail.')


def ReadPts(filename, pointsetlist=None, colorslist=None, merge=True):
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

    :return: The created PointSet or the list of the PointSets

    :rtype: PointSet or list

    """
    colorProp = None
    from tqdm import tqdm

    # Opening file and reading all lines from it
    with open(filename, 'r') as fin:
        lines = fin.readlines()
        batch_start = 0
        batch_size = int(lines[batch_start])
        batch_end = batch_start + batch_size + 1
        batches = []

        while batch_size > 0:

            batch_i = lines[batch_start + 1: batch_end]
            batches.append(batch_i)

            batch_start = batch_end
            try:
                batch_size = int(lines[batch_start])
                batch_end = batch_start + batch_size + 1
            except:
                batch_size = 0
    pts = []
    colors = []
    intensity = []

    for batch in batches:
        pt_batch = (np.array(list(map(__splitPtsString, batch))))
        if merge:
            pts.append(pt_batch[:, :3])
        else:
            try:
                pointsetlist.append(PointSet(pt_batch, path = filename))
            except TypeError:
                print("No pointset list has been assigned.")

        dimension = pt_batch.shape[1]

        if dimension == 7:
            # the data includes RGB
            colors_batch = pt_batch[:, 4:]

            if merge:
                colors.append(colors_batch)
            else:
                try:
                    colorslist.append(ColorFactory.assignColor(pointsetlist[-1], colors_batch))
                except TypeError:
                    print("No colors list has been assigned.")

        if dimension >= 4:
            # the data includes intensity
            intensity_batch = pt_batch[:, 3]

            if merge:
                intensity.append(intensity_batch)
            else:
                try:
                    pointsetlist[-1].AddData2Fields(intensity_batch, field = 'intensity')
                except TypeError:
                    print("No pointset list has been assigned.")

    if merge:
        points = PointSet(np.concatenate(np.array(pts)), path = filename)

        if len(intensity) == len(pts):
            points.AddData2Fields(np.concatenate(np.array(intensity)), field = 'intensity')
        else:
            warnings.warn('Some points don''t have intensity values. None was assigned to PointSet')

        if len(colors) == len(pts):
            if colorslist is not None:
                colorslist.append(ColorFactory.assignColor(points, np.concatenate(np.array(colors))))
        # else:
        #     warnings.warn('Some points don''t have color values. No color property created')

        return points

    else:
        return pointsetlist


def ReadPtx(filename, pointsetlist=list(), colorslist=list(), trasformationMatrices=list(),
            remove_empty=True):
    """
    Reads .ptx file, created by Leica Cyclone

    File is build according to:
    https://w3.leica-geosystems.com/kb/?guid=5532D590-114C-43CD-A55F-FE79E5937CB2

    :param filename: path to file + file

    *Optionals*

    :param pointsetlist: list that holds all the uploaded PointSet
    :param colorslist: list that holds all the color properties that relate to the PointSet
    :param transformationMatrices: list that holds all the transformation properties that relate to the PointSet
    :param removeEmpty: flag to remove or leave empty points. Default: True

    :type filename: str
    :type pointsetlist: list
    :type colorslist: list of ColorProperty
    :type trasnformationMatrices: list of TransformationMatrixProperty
    :type removeEmpty: bool

    :return: pointSet list

    :rtype: list

    .. warning:: Doesn't read the transformation matrices.

    """

    # Open file and read lines from it
    with open(filename) as fin:
        lines = fin.readlines()

        # window size of the scanned points
        batch_start = 0
        num_cols = int(lines[batch_start])
        num_rows = int(lines[batch_start + 1])

        # number of points in the first batch
        batch_size = num_cols * num_rows

        while batch_size > 0:
            # read transformation matrix lines of the current point cloud
            transformationMatrix = np.array(
                [__splitPtsString(line) for line in lines[batch_start + 6:batch_start + 10]])
            # read current point cloud
            pt_batch = np.array([__splitPtsString(line) for line
                                 in lines[batch_start + 10: batch_size + batch_start]])

            # if remove_empty is True - delete all empty points from cloud:
            if remove_empty:
                empty_indices = np.nonzero(pt_batch[:, :3] == np.array([0, 0, 0]))
                pt_batch = np.delete(pt_batch, np.unique(empty_indices), axis = 0)

            pointsetlist.append(PointSet(pt_batch, path = filename, intensity = pt_batch[:, 3]))
            trasformationMatrices.append(TransformationMatrixProperty(pointsetlist[-1],
                                                                      transformationMatrix=transformationMatrix))
            dimension = pt_batch.shape[1]
            if dimension == 7:
                # the data includes RGB
                colors_batch = pt_batch[:, 4:]
                try:
                    colorslist.append(ColorFactory.assignColor(pointsetlist[-1], colors_batch))
                except TypeError:
                    print("No colors list has been assigned.")

            # initialize for next batch.
            batch_start += batch_size + 10
            if batch_start < len(lines) - 10:
                num_cols = int(lines[batch_start])
                num_rows = int(lines[batch_start + 1])

                # number of points in the first batch
                batch_size = num_cols * num_rows
            else:
                batch_size = 0

    return pointsetlist


def ReadXYZ(fileName, pointsetlist=list(), merge=True):
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

    imported_array = np.genfromtxt(fileName, dtype=parametersTypes, filling_values=(0, 0, 0, 0))

    xyz = imported_array[['x', 'y', 'z']].view(float).reshape(len(imported_array), -1)

    pointSet = PointSet(xyz)
    pointSet.path = fileName
    pointsetlist.append(pointSet)

    if merge:
        pointsetlist = np.array(pointsetlist)
    return len(pointsetlist)


def ReadLAS(filename, classification=False):
    """
    Reads LAS or LAZ files

    :param filename: path to LAS or LAZ
    :param classification: return a SegmentationProperty according to the LAS file or not (default: False)

    :type filename: str
    :type classification: bool

    :return: pointcloud, (classification as received from LAS file)
    :rtype: PointSet, SegmentationProperty

    """
    from Segmentation.SegmentationProperty import SegmentationProperty

    with lasfile.File(filename, mode='r') as infile:
        x = infile.X
        y = infile.Y
        z = infile.Z

        pcl = PointSet(np.hstack((x[:, None], y[:, None], z[:, None])), path=filename, intensity=infile.Intensity)

        if classification:
            segmentation = SegmentationProperty(pcl, infile.Classification)
            return pcl, segmentation
        else:
            return pcl


def loadPickle_dataset(filename, type=None):
    """
    Loads a pickle file of a dataset.

    The user can specify which type of data it is .

    :param filename: file and path to file
    :param type: the type of class (BallTreePointSet, PointSet, PointSubSet, etc.)

    :type filename: str
    :type type: class

    :return: BaseData
    """
    import IO_Tools
    filename, extension = IO_Tools.CreateFilename(filename, 'r')

    if type is not None:
        dataset = type()
        dataset.load()
        attrs = _pickle.load(filename)
        print('hello')

def read2_PointSetOpen3D(file_path, voxel_size=-1, print_bb=False):
    '''
    Reads a file into a PointSetOpen3D object

    :param file_path: Path of pointcloud file
    :param voxel_size: If >0 then decimate point cloud with this parameter as min points distance
    :param print_bb: Print boundarybox values

    :type: str
    :type: Positive double
    :type: bool

    :return: pointsetExtra Object
    :rtype: PointSetExtra

    '''
    import sys
    if sys.platform == "linux":
        from PointSetOpen3D import PointSetOpen3D

        # Read Point Cloud
        input_cloud = O3D.read_point_cloud(file_path)

        pointsetExtra = PointSetOpen3D(input_cloud)
        pointsetExtra.DownsampleCloud(voxel_size)
        pointsetExtra.setPath(file_path)
        return pointsetExtra


def GetCurvatureFilePath(folderPath, dataName, currentFileIndex, localNeighborhoodParameters, decimationRadius,
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
    if not testRun:
        r = localNeighborhoodParameters['search_radius']
        nn = localNeighborhoodParameters['maxNN']

        curvatureFilePath = folderPath + 'Curvature/' + dataName + str(currentFileIndex) + 'r' + str(
            r) + "nn" + str(nn)

        if decimationRadius:
            curvatureFilePath += 'd' + str(decimationRadius) + '.txt'
        else:
            curvatureFilePath += '.txt'
    else:
        curvatureFilePath = folderPath + 'Curvature/testRun' + str(currentFileIndex) + '.txt'

    return curvatureFilePath


def rasterFromAscFile(path, projection=None):
    """
    Reads raster from .txt or .asc files

    :param path: path+filename
    :param projection:

    :type path: str

    :return: a RasterData object

     :rtype: RasterData
    """
    from sys import exc_info
    from traceback import print_tb

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

    line_size = len(filelines[6].split(' ')[:-1])
    if line_size == ncols:
        tmp = lambda x: np.float32(x.split(' ')[1:-1])
    else:

        tmp = lambda x: np.float32(x.split(' ')[:-1])

    data = np.array(list(map(tmp, filelines[6:])))
    return RasterData(data, gridSpacing=cellsize, geoTransform=(xllcorner, yllcorner, 1., 1.),
                      spatial_reference=projection, voidData=nodata_value, path=path)

def __splitPtsString(line):
    """
    Extracting the points' data (3D coordinates, intensity, color) from a string

    :param line: A string containing a line from a .pts file

    :return: all existing Data of a point

    :rtype: np.array

    """

    tmp = line.split()
    return np.array(list(map(float, tmp)))


def ReadPly(filename, returnAdditionalAttributes=True):
    r"""
    Reading ply file
    The method returns a PointSet object that contains the 3-D coordinates of all vertices in the ply file and
    their intensity values. If additional attributes exist they are returned as a dictionary with the attribute names
    as the keys

    :param filename: path to \*.ply file
    :param returnAdditionalAttributes: Indicator whether or not return the additional attributes that exist in the file

    :type filename: str
    :type returnAdditionalAttributes: bool

    :return: PointSet object and a dictionary with additional properties (optional)

    :rtype: tuple of a PointSet object and a dictionary
    """

    try:
        plyData = PlyData.read(filename)  # Reading ply file
        properties = list(map(lambda p: p.name, plyData['vertex'].properties))  # Getting list of properties of vertices
        data = plyData['vertex'].data

        # Extracting the 3-D coordinates of the points
        xyz = np.array([data['x'], data['y'], data['z']]).T

        # Extracting the intensity values of the points if they exist
        intensity = data['reflectance'] if 'reflectance' in properties else None

        # Creating the PointSet object
        pntSet = PointSet(points=xyz, intensity=intensity)

        if not returnAdditionalAttributes:
            return pntSet
        else:
            attributes = {}
            for p in properties:
                if p not in ['x', 'y', 'z', 'reflectance']:
                    attributes[p] = data[p]
            return pntSet, attributes
    except:
        print('Failed to extract data from ply file')


