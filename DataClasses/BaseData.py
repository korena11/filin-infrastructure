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


