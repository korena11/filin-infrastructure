from BaseProperty import BaseProperty


class ColorProperty(BaseProperty):
    __rgb = None

    def __init__(self, points, rgb=None):
        """

        :param points:
        :param rgb:

        :type rgb: numpy.array
        """
        super(ColorProperty, self).__init__(points)
        self.load(rgb)  # makes sure that the array is 3d and that it ranges [0,1]

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

    def load(self, colors=None, **kwargs):
        """
        Set
        :param colors:
        :param kwargs:
        :return:
        """
        for key in kwargs:
            if key == 'rgb':
                colors = kwargs[key]
            elif key == '__rgb':
                colors = kwargs[key]
            elif key == 'colors':
                colors = kwargs[key]
            elif key == 'RGB':
                colors = kwargs['RGB']

        if colors is not None:

            import numpy as np
            size_array = colors.shape

            # check the dimension of the array, if only a list of numbers, it should be transformed into a 2D array (nx3)
            if len(size_array) < 2:
                rgb = np.ones((size_array[0], 3))
                rgb *= colors[:, None]
            elif len(size_array) == 3:
                rgb = colors.reshape((size_array[0] * size_array[1], 3))
            else:
                rgb = colors

            # normalize numbers to [0,1] interval
            rgb_normed = (rgb - rgb.min()) / (rgb.max() - rgb.min())

            self.__rgb = rgb_normed

    def getValues(self):
        return self.__rgb

