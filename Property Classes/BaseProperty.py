import h5py

from PointSet import PointSet
from RasterData import RasterData


class BaseProperty(object):
    """
    Base class for all property classes
    """

    def __init__(self, dataset):
        """
        Constructor
        """
        self.__dataset = dataset

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
            raise TypeError('Wrong data type data for this instance')

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
            raise TypeError('Wrong data type data for this instance')

    def save(self, **kwargs):
        r"""
        Save the property in either json or hdf5.

        Default is hdf5.

        .. warning:: Need to be implemented for json

        :param filename: path to filename and filename (with or without extension)
        :param extension: 'h5' or 'json'

        :type filename: str
        :type extension: str

        """
        import re

        filename = kwargs.get('filename', r'output')

        attrs = {}

        # find filename until extension (can get dynamic foldering)
        matched = re.match('(.*)\.([a-z].*)', filename)

        if matched is None:
            # if no extension is in filename, add
            extension = kwargs.get('extension', 'h5')
            filename = filename + '.' + extension
        else:
            # otherwise - use the extension in filename
            extension = matched.group(2)

        if extension == 'h5':
            f = h5py.File(filename, 'w')
            attrs = self.__dict__

            # the group name will be according to the property class name
            property_group = f.create_group(attrs.keys()[0].split('_')[1])

            for key in attrs:
                if len(key.split('__dataset')) > 1:
                    # if it is the dataset attribute - create a subgroup and insert its attributes
                    data_group = property_group.create_group('dataset')
                    dataset_attributes = attrs[key].__dict__
                    for att in dataset_attributes:
                        if dataset_attributes[att] is not None:
                            data_group.create_dataset(key, dataset_attributes[att])

                else:
                    # otherwise - insert the property attributes into an attrs
                    property_group.attrs.create(key, attrs[key])

            f.close()

    # def serialize(self, filename = None):
    #     """
    #     Save property to a json file.
    #
    #     .. warning:: This function may not work. Should be checked
    #
    #     :param filename: path and file name into which the property will be saved.
    #     :type filename: str
    #
    #     """
    #     # TODO: make sure this works.
    #     datasetJson = self.__dataset.serialize()
    #
    #     propertyDict = {
    #         'dataset': datasetJson
    #     }
    #     if filename is None:
    #         return JsonConvertor.serializes(propertyDict)
    #     else:
    #         JsonConvertor.serialize(propertyDict, filename)
    #
    # @classmethod
    # def deserializes(cls, jsonString):
    #     """
    #     Deserialize a JSON object from string
    #
    #     .. warning:: This function may not work. Should be checked
    #
    #     :param jsonString:
    #     :type jsonString: str
    #     :return:
    #     """
    #     tempDictionary = JsonConvertor.deserializes(jsonString)
    #     cls.propertyFromDictionary(tempDictionary)
    #
    # @classmethod
    # def deserialize(cls, filename):
    #     """
    #     Deserialize a JSON object from file
    #
    #     .. warning:: This function may not work. Should be checked
    #
    #     :param filename: The file name and path of the JSON object to be deserialized
    #     :type filename: str
    #
    #     """
    #     tempDictionary = JsonConvertor.deserialize(filename)
    #     cls.propertyFromDictionary(tempDictionary)
    #
    # @classmethod
    # def propertyFromDictionary(cls, data):
    #     """
    #     Assigns the data from json object into the property variables
    #
    #     .. warning:: This function may not work. Should be checked
    #
    #     :param data: dictionary retrieved from json object
    #     :type data: dict
    #
    #     """
    #     s = cls(data['dataset'])

    if __name__ == '__main__':
        import numpy as np

        pts = PointSet(np.array([[1, 1, 1], [2, 1, 1], [3, 1, 1]]))


        # temp = b.serialize()
        # Segment.deserializes(temp)
        # print s.numberOfPatches
        # print s._Segment__patches.keys()
