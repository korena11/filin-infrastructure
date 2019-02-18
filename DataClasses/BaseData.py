'''
infragit
reuma\Reuma
09, May, 2018 
'''

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
        if isinstance(self.__data, BaseData):
            self.__path = self.__data.path

        return self.__path

    @property
    def data(self):
        """
        The data as stored in self.__data
        """
        return self.__data

    @data.setter
    def data(self, data):
        """
        Set data within the data attribute

        :param data of any kind (either points, raster etc.)

        """
        self.__data = data

    @path.setter
    def path(self, path):
        """
        Sets path to dataset

        :param path: path to a file

        :type path: str

        """

        self.__path = path

    # @data.setter
    # def data(self, data):
    #     """
    #     Sets the data to the dataset
    #
    #     :param data of any kind (either points, raster etc.)
    #
    #     """
    #
    #     self.__data = data

    def setValues(self, **kwargs):
        """

        """
        pass

    def save(self, path_or_buf, name='dataset', save_dataset=False):
        """
           Save the PointSet in either json or hdf5.

           Default is hdf5.

           .. warning:: Need to be implemented for json

           :param path_or_buf: the path (string) or file object
           :param extension: 'h5' or 'p' or 'pickle'
           :param group_name: the name for the group that is being saved
           :param save_dataset: flag whether to save the dataset that the property relates to or not. Default: False

           :type path_or_buf: str or h5py.File or file
           :type extension: str
           :type group_name: str
           :type save_dataset: bool

           :return the file after saving
           :rtype: file

           """
        from IO_Tools import CreateFilename
        import SaveFunctions
        import _pickle
        import h5py

        if isinstance(path_or_buf, str):
            path_or_buf, extension = CreateFilename(path_or_buf)

        if isinstance(path_or_buf, h5py.File):
            SaveFunctions.save_dataset_h5(self, path_or_buf, name, save_dataset)

        else:
            try:
                _pickle.dump(self.__dict__, path_or_buf)
            except:
                from warnings import warn

                warn(IOError, 'Not sure how to save')

        return path_or_buf
