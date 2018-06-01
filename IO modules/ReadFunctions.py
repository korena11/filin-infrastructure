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


def __splitPtsString(line):
    """
    Extracting the points' data (3D coordinates, intensity, color) from a string

    :param line: A string containing a line from a .pts file

    :return: all existing Data of a point

    :rtype: np.array

    """
    tmp = line.split()
    return np.array(list(map(float, tmp)))
