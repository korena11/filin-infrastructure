'''
infragit
reuma\Reuma
09, May, 2018 
'''

import re
import warnings

import h5py
import numpy as np

from IO_Tools import CreateFilename


class BaseData(object):
    """
    Base class for all data classes
    """

    def __init__(self):
        """
        Constructor
        """
        self.__path = ''  # Path for the data
        self.__data = None

    @property
    def path(self):
        """
        Path to datafile, if exists

        """
        return self.__path

    @property
    def data(self):
        """
        The data as stored in self.__data
        """
        return self.__data

    def setPath(self, path):
        """
        Sets path to dataset

        :param path: path to a file

        :type path: str

        """

        self.__path = path

    @data.setter
    def data(self, data):
        """
        Sets the data to the dataset

        :param data of any kind (either points, raster etc.)

        """

        self.__data = data

    def setValues(self, **kwargs):
        """

        """
        pass

    def save(self, path_or_buf, **kwargs):
        """
           Save the PointSet in either json or hdf5.

           Default is hdf5.

           .. warning:: Need to be implemented for json

           :param path_or_buf: the path (string) or file object
           :param extension: 'h5' or 'json'
           :param group_name: the name for the group that is being saved
           :param save_dataset: flag whether to save the dataset that the property relates to or not. Default: False

           :type path_or_buf: str or h5py.File
           :type extension: str
           :type group_name: str
           :type save_dataset: bool

           :return the file after

           """

        name = kwargs.get('group_name', 'dataset')
        save_dataset = kwargs.get('save_dataset', False)

        if isinstance(path_or_buf, str):
            path_or_buf, extension = CreateFilename(path_or_buf)

        if isinstance(path_or_buf, h5py.File):
            dataset_attributes = self.__dict__
            data_group = path_or_buf.create_group(name)

            for key in dataset_attributes:
                matched = re.match('\_(.*)\_\_(.*)', key)
                keyname = matched.group(2)

                if keyname == 'data':

                    if len(self.path) == 0:
                        save_dataset = True

                    if not save_dataset:
                        data_group.attrs.create(key, np.string_(self.path))
                        continue

                if dataset_attributes[key] is not None:
                    try:
                        if isinstance(dataset_attributes[key], str):
                            data_group.attrs.create(key, np.string_(dataset_attributes[key]))
                        else:
                            data_group.attrs.create(key, dataset_attributes[key])

                    except:
                        print("{name} attribute will be saved in a differernt group".format(name = keyname))
                        try:
                            dataset_attributes[key].save(path_or_buf, group_name = keyname)
                        except:
                            warn = 'The {a} attribute  was not saved'.format(a = keyname)
                            warnings.warn(warn)
                            continue

        return path_or_buf
