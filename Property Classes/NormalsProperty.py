from BaseProperty import BaseProperty

class NormalsProperty(BaseProperty):
    
    def __init__(self, points, normals):
        super(NormalsProperty, self).__init__(points)
        self.__normals = normals
    
    @property    
    def Normals(self):
        """
        Return points' normals 
        """  
        return self.__normals
    
    @property  
    def dX(self):
        """
        Return normals X coordinates 
        """  
        return self.__normals[:, 0]
    
    @property  
    def dY(self):
        """
        Return normals Y coordinates  
        """  
        return self.__normals[:, 1]
    
    @property  
    def dZ(self):
        """
        Return normals Z coordinates  
        """  
        return self.__normals[:, 2]