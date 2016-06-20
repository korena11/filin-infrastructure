from BaseProperty import BaseProperty

class ColorProperty(BaseProperty):
    
    def __init__(self, points, rgb):
        
        self._BaseProperty__points = points
        self.__rgb = rgb
        
    @property
    def RGB(self):
        return self.__rgb