from BaseProperty import BaseProperty

class CurvatureProperty( BaseProperty ):
    '''
    classdocs
    '''


    def __init__( self, points, curvature ):
        super(CurvatureProperty, self).__init__(points)
        # self._BaseProperty__points = points
        self.__curvature = curvature
    
    @property    
    def Curvature( self ):
        """
        Points' curvature values
        """  
        return self.__curvature
    
    @property  
    def k1( self ):
        """
        Maximal principal curvature value
        """  
        return self.__curvature[:, 0]
    
    @property  
    def k2( self ):
        """
        Minimal principal curvature value
        """  
        return self.__curvature[:, 1]
    
   
