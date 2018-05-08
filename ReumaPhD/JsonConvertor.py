from ast import literal_eval
from json import dumps, dump, load

from numpy import ndarray, array


class JsonConvertor(object):
    @classmethod
    def serializes(cls, obj, *args):
        """
        Serializing an object of any given built-in type to a JSON object
        :param obj: An array to be serialize
        :param filename: The file name and path to save
        """

        if isinstance(obj, (ndarray, list, tuple)):
            if any(map(lambda x: isinstance(x, (tuple, list, ndarray)), obj)):
                temp = map(dumps, map(lambda x: x.tolist() if isinstance(x, (ndarray, tuple)) else x, obj))
            else:
                temp = map(dumps, obj)
            return temp
        # elif isinstance(obj, dict):
        #     if map(lambda k: isinstance(k, int), obj.keys()):
        #         return dumps(dict(zip(map(str, obj.keys()), obj.values())))
        #     else:
        #         return dumps(obj)
        else:
            return dumps(obj)

    @classmethod
    def serialize(cls, obj, filename):
        dump(cls.serializes(obj), open(filename, 'w'))

    @classmethod
    def __deserializes(cls, obj):
        try:
            if isinstance(obj, (ndarray, list, tuple)):
                return array(map(literal_eval, obj))
            elif isinstance(obj, str):
                return literal_eval(obj)
            else:
                return obj
        except ValueError:
            return None

    @classmethod
    def deserializes(cls, obj):
        """
        Deserialize a JSON object from string
        :param obj: The JSON object to be deserialized
        :return: The deserialized object
        """
        jsonObj = literal_eval(obj)
        if isinstance(jsonObj, dict):
            temp = map(cls.__deserializes, jsonObj.values())
            jsonObj.update(dict(zip(jsonObj.keys(), temp)))
        else:
            jsonObj = array(map(cls.__deserializes, jsonObj))

        return jsonObj

    @classmethod
    def deserialize(cls, filename):
        """
        Deserialize a JSON object from file

        :param filename: The file name and path of the JSON object to be deserialized
        :type filename: str

        """
        return cls.deserializes(str(load(open(filename, 'r'))))
