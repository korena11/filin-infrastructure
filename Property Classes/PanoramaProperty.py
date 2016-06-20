from BaseProperty import BaseProperty
from numpy import ones

class PanoramaProperty( BaseProperty ):
    """
    A panoramic representation of the point set
    """
    
    __rowIndexes = None  # An array of indexes corresponding to the row number to which each point belongs to
    __columnIndexes = None  # An array of indexes corresponding to the column number to which each point belongs to
    __dataType = ""  # A string indicating the type of data stored in the panoramic view (e.g. range, intensity, etc.)
    __panoramaData = None  # A m-by-n-by-p array in which the panorama is stored
    __voidData = 250  # A number indicating missing data in the panorama
    __minAzimuth = 0  # The minimal azimuth value
    __maxAzimuth = 360  # The maximal azimuth value
    __minElevation = -45  # The minimal elevation angle value
    __maxElevation = 90  # The maximal elevation angle value
    __azimuthSpacing = 0.057  # The spacing between points in the azimuth direction
    __elevationSpacing = 0.057  # The spacing between points in the elevation angle direction  
    
    
    def __init__( self, points, rowIndexes, columnIndexes, data, **kwargs ):
        """
        Constuctor - Creates a panoramic view of the data sent
        
        :Args:
            - points - PointSet object from which the panorama will be created
            - rowIndexes - The row indexes of the points in the point set based on the elevation angles
            - columnIndexs - The column indexes of the points in the point set based on the azimuth angles
            - data - The data to be represented as a panorama (e.g. range, intesity, etc.)
        """
        self.__rowIndexes = rowIndexes
        self.__columnIndexes = columnIndexes
        self.__points = points
        
        if ( 'dataType' in kwargs.keys() ):
            self.__dataType = kwargs['dataType']
        if ( 'minAzimuth' in kwargs.keys() ):
            self.__minAzimuth = kwargs['minAzimuth']
        if ( 'maxAzimuth' in kwargs.keys() ):
            self.__maxAzimuth = kwargs['maxAzimuth']
        if ( 'minElevation' in kwargs.keys() ):
            self.__minElevation = kwargs['minElevation']
        if ( 'maxElevation' in kwargs.keys() ):
            self.__maxElevation = kwargs['maxElevation']
        if ( 'azimuthSpacing' in kwargs.keys() ):
            self.__azimuthSpacing = kwargs['azimuthSpacing']
        if ( 'elevationSpacing' in kwargs.keys() ):
            self.__elevationSpacing = kwargs['elevationSpacing']
        
        numRows = int( ( self.__maxElevation - self.__minElevation ) / self.__elevationSpacing ) + 1
        numColumns = int( ( self.__maxAzimuth - self.__minAzimuth ) / self.__azimuthSpacing ) + 1
        
        if ( len( data.shape ) == 1 ):
            self.__panoramaData = self.__voidData * ones( ( numRows, numColumns ) )
            self.__panoramaData[rowIndexes, columnIndexes] = data
        else:
            self.__panoramaData = self.__voidData * ones( ( numRows, numColumns, data.shape[1] ) )
            self.__panoramaData[rowIndexes, columnIndexes, :] = data[:, :]
            
    @property
    def PanoramaImage( self ):
        """
        Returns the panorama image
        """
        return self.__panoramaData
