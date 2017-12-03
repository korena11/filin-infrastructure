from PointSet import PointSet
from RasterData import RasterData

class BaseProperty(object):
    '''
    Base class for all property classes
    '''

    def __init__(self, dataset):
        '''
        Constructor
        '''
        if isinstance(dataset, PointSet):
            self.__points = dataset

        elif isinstance(dataset, RasterData):
            self.__raster = dataset
    
    @property   
    def Points(self):
        '''
        Returns the point set
        '''
        return self.__points

    @property
    def Raster(self):
        return self.__raster