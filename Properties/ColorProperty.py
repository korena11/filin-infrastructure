from BaseProperty import BaseProperty


class ColorProperty(BaseProperty):
    __rgb = None

    def __init__(self, points, rgb):
        """

        :param points:
        :param rgb:

        :type rgb: numpy.array
        """
        super(ColorProperty, self).__init__(points)
        self.setValues(rgb)
        
    @property
    def RGB(self):
        """
        Points' colors, ranging 0-255 (integers)

        :return: rgb

        :rtype: int, nx3 numpy.array

        """

        RGB = (self.__rgb * 255).astype(int)

        return RGB

    @property
    def rgb(self):
        """
        Points' colors, ranging 0-1 (floats)

        :return: rgb

        :rtype: float, nx3 numpy.array
        """

        return self.__rgb

    def setValues(self, *args, **kwargs):
        """
        Set
        :param args:
        :param kwargs:
        :return:
        """

        # normalize numbers to [0,1] interval
        rgb = args[0]
        rgb_normed = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        self.__rgb = rgb_normed

    def getValues(self):
        return self.__rgb
