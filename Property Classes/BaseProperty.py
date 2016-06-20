class BaseProperty:
    '''
    Base class for all property classes
    '''

    def __init__(self, points):
        '''
        Constructor
        '''
        self.__points = points
    
    @property   
    def Points(self):
        '''
        Returns the point set
        '''
        return self.__points
