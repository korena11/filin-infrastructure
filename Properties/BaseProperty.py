from IO_Tools import CreateFilename
from PointSet import PointSet
from RasterData import RasterData


class BaseProperty(object):
    """
    Base class for all property classes. All properties are iterable, however the values the iteration returns need to
    be defined for each property individually as methods: :meth:`__getPointProperty` and :meth:`__setPointProperty`. These functions
    are usually called from another function, unique for each property to minimize confusion.

    """

    def __init__(self, dataset = None):
        """
        Constructor
        """
        self.__dataset = dataset
        # --------- To make the object iterable ---------
        self.current = 0

        # ---------- Definitions to make iterable -----------

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        try:
            return self.__getPointProperty(self.current - 1)
        except IndexError:
            self.current = 0
            raise StopIteration

    def __reset__(self):
        """
        Reset iterable
        :return:
        """
        self.__current = 0

    def __getPointProperty(self, idx):
        """
        Retrieve the point property (object or value) of a specific point

        :param idx: the point index

        :return: principal curvature values (k1, k2)

        :rtype: float

        .. warning::
            This function needs to be overwritten for each inheriting property. It is empty at the BaseProperty


        """
        pass

    @property
    def Points(self):
        """
        Holds the points dataset

        :return: PointSet related to this property
        :rtype: PointSet

        """
        if isinstance(self.__dataset, PointSet):
            return self.__dataset
        else:
            # warn('Wrong data type data for this instance')
            return False

    @property
    def Raster(self):
        """
        Holds raster dataset

        :return: RasterData related to this property
        :rtype: RasterData

        """
        if isinstance(self.__dataset, RasterData):
            return self.__dataset
        else:
            # warn('Wrong data type data for this instance')
            return False

    def setValues(self, *args, **kwargs):
        """
        Sets the all values of a property.

        :param args: according to the property
        :param kwargs: according to the property

        .. warning::
            This function needs to be overwritten for each inheriting property. It is empty at the BaseProperty

        """
        pass

    @property
    def path(self):
        """
        The path of the dataset

        :return: the path of the dataset

        :rtype: str
        """
        return self.__dataset.path

    @property
    def Size(self):
        """
        Return the size of the property
        """
        if isinstance(self.__dataset, RasterData):
            return self.__dataset.cols * self.__dataset.rows

        elif isinstance(self.__dataset, PointSet):
            return self.__dataset.Size

    def getValues(self):
        """
        Get a specific characteristic value of the property

        :return: value for each point

        :rtype: np.array

        .. warning::
            This function needs to be overwritten for each inheriting property. It is empty at the BaseProperty

        """
        pass

    def save(self, filename, save_dataset=False):
        """
        Save the property in either json or hdf5.

        Default is hdf5.

        .. warning:: Need to be implemented for json

        :param filename: can be a filename or a path and filename, with or without extension.
        :param save_dataset: flag whether to save the dataset that the property relates to or not. Default: False

        :type filename: str
        :type save_dataset: bool

        """
        import SaveFunctions
        import h5py
        attrs = {}

        if isinstance(filename, str):
            filename, extension = CreateFilename(filename)

        if isinstance(filename, h5py.File):
            SaveFunctions.save_property_h5(self, filename, save_dataset)

        else:
            try:
                SaveFunctions.pickleProperty(self, filename, save_dataset)
            except:
                from warnings import warn
                warn(IOError, 'Not sure how to save')


    if __name__ == '__main__':
        import numpy as np

        pts = PointSet(np.array([[1, 1, 1], [2, 1, 1], [3, 1, 1]]))


        # temp = b.serialize()
        # Segment.deserializes(temp)
        # print s.numberOfPatches
        # print s._Segment__patches.keys()
