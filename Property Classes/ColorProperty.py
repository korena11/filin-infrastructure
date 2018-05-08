from BaseProperty import BaseProperty

class ColorProperty(BaseProperty):
    
    def __init__(self, points, rgb):
        super(ColorProperty, self).__init__(points)
        self.__rgb = rgb
        
    @property
    def RGB(self):
        """
        Points' colors in rgb

        :return: rgb

        :rtype: nx3 nd-array

        """
        return self.__rgb
