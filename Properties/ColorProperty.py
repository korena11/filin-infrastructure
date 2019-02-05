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
        self.setValues(rgb)  # makes sure that the array is 3d and that it ranges [0,1]

    @property
    def RGB(self):
        """
        Points' colors, ranging 0-255 (integers)

        :return: rgb

        :rtype: int, nx3 numpy.array

        """

        RGB = (self.rgb * 255).astype(int)

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

        array = args[0]

        import numpy as np
        size_array = array.shape

        # check the dimension of the array, if only a list of numbers, it should be transformed into a 2D array (nx3)
        if len(size_array) < 2:
            rgb = np.ones((size_array[0], 3))
            rgb *= array[:, None]
        elif len(size_array) == 3:
            rgb = array.reshape((size_array[0] * size_array[1], 3))

        else:
            rgb = array
        # normalize numbers to [0,1] interval

        rgb_normed = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        self.__rgb = rgb_normed

    def getValues(self):
        return self.__rgb

