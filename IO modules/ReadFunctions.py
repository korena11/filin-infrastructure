"""
Reading Functions
=================

Specific functions for file reading. Called from the IOFactory

"""

import warnings

# from laspy.file import file
import numpy as np

from ColorFactory import ColorFactory
from PointSet import PointSet
from TransformationMatrixProperty import TransformationMatrixProperty


def ReadPts(filename, pointsetlist=[], colorslist=[], merge=True):
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
            points.AddData2Fields(np.array(intensity)[0], field = 'intensity')
        else:
            warnings.warn('Some points don''t have intensity values. None was assigned to PointSet')

        if len(colors) == len(pts):
            colorslist = ColorFactory.assignColor(points, np.array(colors)[0])
        else:
            warnings.warn('Some points don''t have color values. No color property created')

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


def ReadLAS(filename):
    """
    .. warning:: Not implemented.

    :param filename:

    :return:

    """
    pass


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
        import open3d as O3D
        from PointSetOpen3D import PointSetOpen3D

        # Read Point Cloud
        input_cloud = O3D.read_point_cloud(file_path)

        pointsetExtra = PointSetOpen3D(input_cloud)
        pointsetExtra.DownsampleCloud(voxel_size)

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
        r = localNeighborhoodParameters['r']
        nn = localNeighborhoodParameters['nn']

        curvatureFilePath = folderPath + 'Curvature/' + dataName + str(currentFileIndex) + 'r' + str(
            r) + "nn" + str(nn)

        if decimationRadius:
            curvatureFilePath += 'd' + str(decimationRadius) + '.txt'
        else:
            curvatureFilePath += '.txt'
    else:
        curvatureFilePath = folderPath + 'Curvature/testRun' + str(currentFileIndex) + '.txt'

    return curvatureFilePath


def __splitPtsString(line):
    """
    Extracting the points' data (3D coordinates, intensity, color) from a string

    :param line: A string containing a line from a .pts file

    :return: all existing Data of a point

    :rtype: np.array

    """

    tmp = line.split()
    return np.array(list(map(float, tmp)))
