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

    def load(self, **kwargs):
        """
        Loads all values of a property.

        :param kwargs: according to the property

        In general, sets every attribute if it exists within the property, and throws a warning if it is not
        within the property

        """
        for key in kwargs:
            try:
                self.__getattribute__(key)
                self.__setattr__(key, kwargs[key])
            except AttributeError:
                key_modified = '_' + self.__class__.__name__ + '__' + key
                self.__setattr__(key_modified, kwargs[key])

            except TypeError:
                from warnings import warn
                warn('No attribute by the name of %s' % (key))
                continue

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

        :rtype: int
        """
        if isinstance(self.__dataset, RasterData):
            return int(self.__dataset.cols * self.__dataset.rows)

        elif isinstance(self.__dataset, PointSet):
            return int(self.__dataset.Size)

    def getValues(self):
        """
        Get a specific characteristic value of the property

        :return: value for each point

        :rtype: np.array

        .. warning::
            This function needs to be overwritten for each inheriting property. It is empty at the BaseProperty

        """
        pass




    if __name__ == '__main__':
        import numpy as np

        pts = PointSet(np.array([[1, 1, 1], [2, 1, 1], [3, 1, 1]]))


        # temp = b.serialize()
        # Segment.deserializes(temp)
        # print s.numberOfPatches
        # print s._Segment__patches.keys()
