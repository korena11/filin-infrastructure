# Updated Last on 13/06/2014 14:07
 
import numpy as np
from string import split
from numpy import array, asarray, hstack, tile, ndarray, where, savetxt
from PointSet import PointSet
from ColorProperty import ColorProperty
from SegmentationProperty import SegmentationProperty
from SphericalCoordinatesProperty import SphericalCoordinatesProperty

from shapefile import Writer, Reader, POINTZ
import linecache

# splitline = lambda line : [float(x) for x in split(line, ' ')]
# numpy.array(map(splitline,file))



# Class for loading and saving point clouds from files
class IOFactory:
    """
    Create PointSet from different types of input files: .pts, .xyz
    
    Write PointSet Data to different types of file: .pts(TODO), .xyz(TODO), shapeFile
    """

    # ---------------------------READ -------------------------------
    @classmethod    
    def ReadPts( cls, filename, pointsetlist, **kwargs ):
        """
        Reading points from *.pts file
        Creates one or more PointSet objects, returned through pointSetList 
        

        :param fileName (str): name of .pts file
        :param pointSetList (list): place holder for created PointSet objects
        :param merge (boolean) : True to merge points in file; False do no merge
            

        :return int.  Number of PointSet objects created
        """
        
        # Opening file and reading all lines from it
        fin = open( filename )
        fileLines = fin.read()
        fin.close()
        
        
        # Splitting into list of lines 
        lines = split( fileLines, '\n' )
        del fileLines
                    
                    
        # Removing the last line if it is empty
        while( True ):
            numLines = len( lines )
            if lines[numLines - 1] == "":
                numLines -= 1
                lines = lines[0: numLines]
            else:
                break
                            
        # Removing header line
        readLines = 0        
        while readLines < numLines:
            data = []
            # Read number if points in current cloud
            numPoints = int( lines[readLines] )
            # Read the points
            currentLines = lines[readLines + 1: readLines + numPoints + 1]                                                                                    
            # Converting lines to 3D Cartesian coordinates Data                
            data = map( cls.__splitPtsString, currentLines )
            # advance lines counter
            readLines += numPoints + 1

            data = array( data )          
            
            xyz = asarray( data[:, 0:3] )
            rgb = None
            intensity = None
            
            if numPoints == 1:           
                kwargs['vertList'].append( xyz )
            else:
                if data.shape[1] == 6:
                    rgb = np.asarray( data[:, 3:6], dtype = np.uint8 )
                if data.shape[1] == 7:
                    rgb = np.asarray( data[:, 4:7], dtype = np.uint8 )            
                if data.shape[1] == 4 or data.shape[1] == 7:
                    intensity = np.asarray( data[:, 3], dtype = np.int )         
                # Create the PointSet object
                pointSet = PointSet( xyz, rgb, intensity )            
                pointsetlist.append(pointSet)
                
        del lines

        if kwargs.get('merge', True):
                return cls.__mergePntList(pointSetList)
        else:
            return len( pointsetlist )
    
    
    @classmethod    
    def ReadPtx( cls, fileName, pointSetList ):
        """
        Reading points from *.pts file
        Creates one or more PointSet objects, returned through pointSetList 
        
        :Args:
            fileName (str): name of .pts file
            pointSetList (list): place holder for created PointSet objects            
            
        :Returns:
             int.  Number of PointSet objects created
 
        """
        
        # Opening file and reading all lines from it
        fin = open( fileName )
        fileLines = fin.read()
        fin.close()
        
        
        # Splitting into list of lines 
        lines = split( fileLines, '\n' )
        del fileLines
                    
                    
        # Removing the last line if it is empty
        while( True ):
            numLines = len( lines )
            if lines[numLines - 1] == "":
                numLines -= 1
                lines = lines[0: numLines]
            else:
                break
                            
        # Removing header line
        data = []
#         currentLines = lines[10::]                                                                                    
        # Converting lines to 3D Cartesian coordinates Data     
        linesLen = map( lambda x: len( x ), lines )   
        line2del = ( where( np.asarray( linesLen ) < 5 )[0] )
        if len( line2del ) > 1 and line2del[0] - line2del[1] == -1:
            line2del = line2del[-2::-2]  # there two lines one after another with length 1, we need the first one
        for i2del in line2del:
            del lines[i2del:i2del + 10] 
        data = map( cls.__splitPtsString, lines )
        line2del = where( asarray( data )[:, 0:4] == [0, 0, 0, 0.5] )[0]
        data = np.delete( data, line2del, 0 )         

        data = array( data )          
        
        xyz = asarray( data[:, 0:3] )
        if data.shape[1] == 6:
            rgb = np.asarray( data[:, 3:6], dtype = np.uint8 )
            pointSet = PointSet( xyz, rgb )
        if data.shape[1] == 7:
            rgb = np.asarray( data[:, 4:7], dtype = np.uint8 ) 
            intensity = np.asarray( data[:, 3], dtype = np.int )  
            pointSet = PointSet( xyz, rgb, intensity )           
        if data.shape[1] == 4 or data.shape[1] == 7:
            intensity = np.asarray( data[:, 3], dtype = np.int )  
            pointSet = PointSet( xyz, intensity )
                   
        # Create the List of PointSet object            
        pointSetList.append( pointSet ) 
        
        del lines
        
        return len( pointSetList )
    
    
    @classmethod    
    def ReadXYZ( cls, fileName, pointSetList ):
        """
        Reading points from *.xyz file
        Creates one PointSet objects returned through pointSetList
        
        Args:
            fileName (str): name of .xyz file            
            pointSetList (list): place holder for created PointSet object
            
        Returns:
           int.  Number of PointSet objects created
 
        """                 
        parametersTypes = np.dtype( {'names':['name', 'x', 'y', 'z']
                               , 'formats':['int', 'float', 'float', 'float']} )
        
#         parametersTypes = np.dtype({'names':['x', 'y', 'z']
#                                , 'formats':['float', 'float', 'float']})
            
        imported_array = np.genfromtxt( fileName, dtype = parametersTypes, filling_values = ( 0, 0, 0, 0 ) )
    
        xyz = imported_array[['x', 'y', 'z']].view( float ).reshape( len( imported_array ), -1 )
                            
        pointSet = PointSet( xyz )
        pointSetList.append( pointSet )
        
        return len( pointSetList )

    @classmethod
    def ReadShapeFile(cls, fileName, pointSetList):
        """
         Importing points from shapefiles
         :Todo:
             - make small functions from other kinds of shapefiles rather than polylines

         :Args:
             fileName (str): Full path and name of the shapefile to be created (not including the extension)
             pointSetList (list): place holder for created PointSet objects
         """
        shape = Reader(fileName)

        if shape.shapeType == 3:
            shapeRecord = shape.shapeRecords()
            polylinePoints = map(cls.__ConvertRecodrsToPoints, shapeRecord)

            for i in (polylinePoints):
                pointSet = PointSet(i)
                pointSetList.append(pointSet)


        else:
            return 0

    def read_ascii_grid(cls, filename, **kwargs):
        """

        :param filename: .asc filename
        :param raster: rasterProperty variable, will hold the spatial information data
        :return: raster numpy array

        """
        #:TODO CHECK RUNNING (wasn't debugged)

        myArray = np.loadtxt(filename, skiprows=6)

        for i in range(0,6):
            line = linecache.getline(filename, i)
            spatialData = cls.__splitPtsString(line)[1]

        ncolmns = spatialData[0]
        nrows = spatialData[1]
        xll = spatialData[2]
        yll = spatialData[3]
        cellSize = spatialData[4]
        nodata_value = spatialData[5]

        if ('raster' in kwargs.keys()):
            kwargs['raster'].setValues(ncolumns=ncolmns, nrows=nrows, extent={'east': xll, 'south': yll},
                                       cellSize=cellSize, noDataValue=nodata_value)


    # ---------------------------WRITE -------------------------------
    @classmethod
    def WriteToPts( cls, points, path ):
        '''
        Write to pts file
        :Args:
            - points: PointSet
            - path: path to the directory of a new file + file name
        '''
        
        fields_num = points.FieldsDimension
        if fields_num == 7:
            data = hstack( ( points.ToNumpy(), points.Intensity, points.RGB ) )
            fmt = ['%.3f', '%.3f', '%.3f', '%d', '%d', '%d', '%d']
        elif fields_num == 6:
            data = hstack( ( points.ToNumpy(), points.RGB ) ) 
            fmt = ['%.3f', '%.3f', '%.3f', '%d', '%d', '%d']
        elif fields_num == 4:
            data = hstack( ( points.ToNumpy(), points.Intensity ) )
            fmt = ['%.3f', '%.3f', '%.3f', '%d']
        else:
            data = points.ToNumpy()
            fmt = ['%.3f', '%.3f', '%.3f']
            
        savetxt( path, points.Size, fmt = '%long' )
        with open( path, 'a' ) as f_handle:
            savetxt( f_handle, data, fmt, delimiter = '\t', newline = '\n' )

    @classmethod
    def WriteToShapeFile( cls, pointSet, fileName, **kwargs ):
        """
        Exporting points to shapefile
        
        :Args:
            pointSet: A PointSet\PointSubSet object with the points to be extracted
            fileName: Full path and name of the shapefile to be created (not including the extension)
            Additional properties can be sent using **kwargs which will be added as attributes in the shapfile
        """
        if ( pointSet.Z != None ):
            fieldList = ['X', 'Y', 'Z']
        else:
            fieldList = ['X', 'Y']
        
        attributes = pointSet.ToNumpy
        
        if ( pointSet.Intensity != None ):
            fieldList.append( 'intensity' )
            attributes = hstack( [attributes, pointSet.Intensity.reshape( ( pointSet.Size, 1 ) )] )
        if ( pointSet.RGB != None ):
            fieldList.append( 'r' )
            fieldList.append( 'g' )
            fieldList.append( 'b' )
            attributes = hstack( [attributes, pointSet.RGB] )
            
        for auxPropertyName, auxProperty in kwargs.iteritems():
            if ( isinstance( auxProperty, ColorProperty ) ):
                fieldList.append( auxPropertyName + '_r' )
                fieldList.append( auxPropertyName + '_g' )
                fieldList.append( auxPropertyName + '_b' )
                attributes = hstack( [attributes, pointSet.RGB] )
            elif ( isinstance( auxProperty, SegmentationProperty ) ):
                fieldList.append( 'labels_' + auxPropertyName )
                attributes = hstack( [attributes,
                                     auxProperty.GetAllSegments.reshape( ( pointSet.Size, 1 ) )] )
            elif ( isinstance( auxProperty, SphericalCoordinatesProperty ) ):
                fieldList.append( 'azimuth' )
                fieldList.append( 'elevationAngle' )
                fieldList.append( 'Range' )
                attributes = hstack( [attributes, auxProperty.ToNumpy] )

        
        w = Writer( POINTZ )
        
        map( w.field, fieldList, tile( 'F', len( fieldList ) ) )
        if ( pointSet.Z != None ):
            map( w.point, pointSet.X, pointSet.Y, pointSet.Z )
            
        else:
            map( w.point, pointSet.X, pointSet.Y )
                    
#         map(w.record, attributes2)
        w.records = map( ndarray.tolist, attributes )
        
        w.save( fileName )

    # ---------------------------PRIVATES -------------------------------
    @classmethod
    def __ConvertRecodrsToPoints( cls, shapeRecord ):
        """ 
        Converting polyline into points
        
        :Args:
            - fileName: Full path and name of the shapefile to be created (not including the extension)
        """
        
        points = array( shapeRecord.shape.points )
        return points


    @classmethod
    def __mergePntList(cls, pointSetList):
        '''
        Merging several pointset

        :Args:
            - pointSetList: a list of PointSets
        :Returns:
            - refPntSet: merged PointSet
        '''
        # TODO: changed from MegrePntList to __MergePntList. CHECK WHERE WAS IN USE!

        list_length = len(pointSetList)
        if list_length > 1:
            refPntSet = pointSetList[0]
            fields_num = refPntSet.FieldsDimension
            for pntSet in pointSetList[1::]:
                if fields_num == 7:
                    refPntSet.AddData2Fields(pntSet.ToNumpy(), field='XYZ')
                    refPntSet.AddData2Fields(pntSet.RGB, field='RGB')
                    refPntSet.AddData2Fields(pntSet.Intensity, field='Intensity')
                elif fields_num == 6:
                    refPntSet.AddData2Fields(pntSet.ToNumpy(), field='XYZ')
                    refPntSet.AddData2Fields(pntSet.RGB, field='RGB')
                elif fields_num == 4:
                    refPntSet.AddData2Fields(pntSet.ToNumpy(), field='XYZ')
                    refPntSet.AddData2Fields(pntSet.Intensity, field='Intensity')
                else:
                    refPntSet.AddData2Fields(pntSet.ToNumpy(), field='XYZ')

            return refPntSet
        else:
            return pointSetList[0]

    @classmethod
    def __splitPtsString(cls, line):
        """
        Extracting from a string the Data of a point (3D coordinates, intensity, color)

        :Args:
            - line: A string containing a line from a *.pts file

        :Returns:
            -  An array containing all existing Data of a point
        """
        tmp = split(line, ' ')
        return map(float, tmp)

        # return [float(x) for x in split(line, ' ')]
     
        
     
        
if __name__ == "__main__":
    
    pointSetList = []
    #====================Point cloud test=================================
    # fileName = '..\Sample Data\Geranium2Clouds.pts'
    # IOFactory.ReadPts(fileName, pointSetList)
    #===========================================================================
    
    #=================shapefile test==================================
    polylineFileName = r'E:\My Documents\Projects\IOLR\HaBonim\polyRimSample.shp'
    IOFactory.ReadShapeFile( polylineFileName, pointSetList )
   
    
