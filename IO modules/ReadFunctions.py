"""
Functions for reading files
"""

import warnings

import numpy as np

from ColorFactory import ColorFactory
from PointSet import PointSet


def ReadPts(filename, *args, **kwargs):
    """
    Reading points from .pts file. If the pts file holds more than one PointSet merge into one PointSet (unless told
    otherwise).

    :param fileName: name of .pts file

    :param pointsetlist: placeholder for created PointSet objects
    :param colorslist: placeholder for ColorProperty for PointSet object(s)


    **Optionals**

    :param merge: merge points in file into one PointSet or not. Default: True.

    :type filename: str
    :type pointsetlist: list
    :type colorslist: list
    :type merge: bool

    :return: The created PointSet or the list of the PointSets created and the ColorProperty that belongs to it

    :rtype: tuple

    """
    colorProp = None

    if args:
        pointsetlist = args[0]
        colorslist = args[1]

    merge = kwargs.get('merge', True)

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
        points = PointSet(np.array(pts)[0], path = filename)

        if len(intensity) == len(pts):
            points.AddData2Fields(np.array(intensity)[0], field = 'intensity')
        else:
            warnings.warn('Some points don''t have intensity values. None was assigned to PointSet')

        if len(colors) == len(pts):
            colorProp = ColorFactory.assignColor(points, np.array(colors)[0])
        else:
            warnings.warn('Some points don''t have color values. No color property created')

        return points, colorProp

    else:
        return pointsetlist, colorslist


def ReadPtx(filename, pointsetlist = None):
    """
    Reads .ptx file, created by Leica Cyclone

    File is build according to:
    https://w3.leica-geosystems.com/kb/?guid=5532D590-114C-43CD-A55F-FE79E5937CB2

    :param filename: path to file + file
    :param pointsetlist: list that holds all the uploaded PointSet

    :type filename: str
    :type pointsetlist: list

    :return: pointSet list

    :rtype: list

    .. warning:: Doesn't read the transformation matrices.
    """

    # Opening file and reading all lines from it
    with open(filename) as fin:
        lines = fin.readlines()

        # window size of the scanned points
        batch_start = 0
        num_cols = int(lines[batch_start])
        num_rows = int(lines[batch_start + 1])

        # number of points in the first batch
        batch_size = num_cols * num_rows

        while batch_size > 0:
            transformationMatrix = np.array(float(lines[batch_start + 6: batch_start + 10]))
            print('hello')

    # Removing header line
    data = []
    #         currentLines = lines[10::]
    # Converting lines to 3D Cartesian coordinates Data
    linesLen = [len(x) for x in lines]
    line2del = (np.where(np.asarray(linesLen) < 5)[0])

    if len(line2del) > 1 and line2del[0] - line2del[1] == -1:
        line2del = line2del[-2::-2]  # there two lines one after another with length 1, we need the first one
    for i2del in line2del:
        del lines[i2del:i2del + 10]
    data = list(map(__splitPtsString, lines))
    line2del = np.where(np.asarray(data)[:, 0:4] == [0, 0, 0, 0.5])[0]
    data = np.delete(data, line2del, 0)

    data = np.array(data)

    xyz = np.asarray(data[:, 0:3])
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

    pointSet.setPath(filename)
    # Create the List of PointSet object
    pointsetlist.append(pointSet)

    del lines

    return len(pointsetlist)

def __splitPtsString(line):
    """
    Extracting the points' data (3D coordinates, intensity, color) from a string

    :param line: A string containing a line from a .pts file

    :return: all existing Data of a point

    :rtype: np.array

    """
    tmp = line.split()
    return np.array(list(map(float, tmp)))
