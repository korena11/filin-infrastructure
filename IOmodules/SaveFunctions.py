"""
Saving Functions
=================

Specific functions for file saving. Called from the ``IOFactory``

"""
import sys

if sys.version_info[0] < 3:
    import cPickle as _pickle
else:
    import _pickle

import re
import warnings

import h5py
import numpy as np
from numpy import hstack, tile, ndarray, savetxt

from BaseData import BaseData
from BaseProperty import BaseProperty
from Color.ColorProperty import ColorProperty
from PointSet import PointSet
from PointSubSet import PointSubSet
from SegmentationProperty import SegmentationProperty
from SphericalCoordinatesProperty import SphericalCoordinatesProperty
from shapefile import Writer, POINTZ

def WriteToPts(points, path):
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

    savetxt(path, points.Size, fmt='%long')
    with open(path, 'a') as f_handle:
        savetxt(f_handle, data, fmt, delimiter='\t', newline='\n')


def WriteToShapeFile(pointSet, fileName, colors=None, **kwargs):
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


def save_property_h5(property, h5file, save_dataset=False):
    """
    Save property to hdf5

    :param property: the proeprty to save
    :param h5file: file object of hdf5
    :param save_dataset: flag whether to save the dataset that the property relates to or not. Default: False

    :type property: BaseProperty
    :type h5file: h5py.file
    :type save_dataset: bool

    """
    attrs = property.__dict__

    # the group name will be according to the property class name
    groupname = list(attrs.keys())[0].split('_')[1]
    property_group = h5file.create_group(groupname)

    for key in attrs:
        if len(key.split('__dataset')) > 1:
            # if it is the dataset attribute - create a subgroup and insert its attributes
            save_property_h5(property.__dataset, h5file, name='_' + groupname + '__dataset',
                             save_dataset=save_dataset)
        else:
            # otherwise - insert the property attributes into an attrs
            property_group.attrs.create(key, attrs[key])


def save_dataset_h5(dataset, h5file, name='dataset', save_dataset=False):
    """
    :param dataset: dataset to save
    :param h5file: file object of hdf5
    :param name: name of the group for the dataset (default: 'dataset')
    :param save_dataset: flag whether to save the dataset that the property relates to or not. Default: False

    :type dataset: BaseData
    :type h5file: h5py.File
    :type name: str
    :type save_dataset: bool

        """
    dataset_attributes = dataset.__dict__
    data_group = h5file.create_group(name)

    for key in dataset_attributes:
        matched = re.match('\_(.*)\_\_(.*)', key)
        if matched is None:
            continue
        keyname = matched.group(2)

        if keyname == 'data':

            if len(dataset.path) == 0:
                save_dataset = True

            if not save_dataset:
                data_group.attrs.create(key, np.string_(dataset.path))
                continue

        if dataset_attributes[key] is not None:
            try:
                if isinstance(dataset_attributes[key], str):
                    data_group.attrs.create(key, np.string_(dataset_attributes[key]))
                else:
                    data_group.attrs.create(key, dataset_attributes[key])

            except:
                print("{name} attribute will be saved in a different group".format(name=keyname))
                try:
                    dataset_attributes[key].save(h5file, group_name=keyname)
                except:
                    warn = 'The {a} attribute  was not saved'.format(a=keyname)
                    warnings.warn(warn)
                    continue


def pickleProperty(property, fileobj, save_dataset=False):
    """
    Saves a property to a filename

    :param property: The property to save
    :param fileobj: The filename to which the property will be saved
    :param save_dataset: flag whether to save the dataset that the property relates to or not. Default: False

    :type property: BaseProperty
    :type fileobj: file
    :type save_dataset: bool

    :return: success of failure
    :rtype: bool

    .. warning::

        Does not work for properties that hold PointSetOpen3D
    """
    filename = fileobj.name
    attrs = property.__dict__
    att_types = [(att, type(property.__getattribute__(att))) for att in attrs]

    if '_BaseProperty__dataset' in attrs:
        dataset = attrs.pop('_BaseProperty__dataset')
        attrs.update({'_BaseProperty__path': dataset.path})
        if isinstance(dataset, PointSubSet):
            attrs.update({'indices': dataset.indices})
    if 'current' in attrs:
        attrs.pop('current')

    # for att, att_type in att_types:
    #     if issubclass(att_type, BaseProperty):
    #         pickleProperty(property.__getattribute__(att), )

    _pickle.dump(attrs, fileobj)
    fileobj.close()

    # if the dataset is to be saved
    if save_dataset:
        name, extension = filename.split('.')
        data_name = filename.split('.')[0] + '__data.' + extension  # it will be saved to a different file
        datafile = dataset.save(data_name)
        datafile.close()


def pickleDataset(dataset, path_or_buf, **kwargs):
    """
       Save the PointSet in either json or hdf5.

       Default is hdf5.

       .. warning:: Need to be implemented for json

       :param dataset: the dataset to save
       :param path_or_buf: the path (string) or file object
       :param extension: 'h5' or 'p' or 'pickle'

       :type dataset: BaseData or subclass
       :type path_or_buf: str or h5py.File or file
       :type extension: str

       :return the file after saving
       :rtype: file

       """
    from IO_Tools import CreateFilename
    import _pickle

    if isinstance(path_or_buf, str):
        path_or_buf, extension = CreateFilename(path_or_buf)

    try:
        _pickle.dump(dataset, path_or_buf)
    except:
        from warnings import warn

    return path_or_buf
