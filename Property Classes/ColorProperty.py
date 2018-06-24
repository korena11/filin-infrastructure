from BaseProperty import BaseProperty


class ColorProperty(BaseProperty):
    __rgb = None

    def __init__(self, points, rgb):
        super(ColorProperty, self).__init__(points)
        self.setValues(rgb)
        
    @property
    def RGB(self):
        """
        Points' colors in rgb

        :return: rgb

        :rtype: nx3 nd-array

        """
        return self.__rgb

    def setValues(self, *args, **kwargs):
        """
        Set
        :param args:
        :param kwargs:
        :return:
        """
        self.__rgb = args[0]

    def getValues(self):
        return self.__rgb
