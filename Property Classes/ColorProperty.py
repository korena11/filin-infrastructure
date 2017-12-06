from BaseProperty import BaseProperty

class ColorProperty(BaseProperty):
    
    def __init__(self, points, rgb):
        super(ColorProperty, self).__init__(points)
        self.__rgb = rgb
        
    @property
    def RGB(self):
        return self.__rgb