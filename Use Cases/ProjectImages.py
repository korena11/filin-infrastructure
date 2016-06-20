from matplotlib.image import imread # Read image
import numpy as np # Numpy arrays
from string import split # Used in matrix file reading
from PointSet import PointSet
from ColorFactory import ColorFactory
from IOFactory import IOFactory
from Visualization import Visualization 


def ReadProjectionMatrixFromFile(fileName):
    
    fin = open(fileName)
    fileLines = fin.read()   # Reading all the file
    fin.close()
    
    # Split the text into lines
    fileLines = split(fileLines, '\n')

    # Defining the transform matrix of the current image
    transformMatrix = np.matrix(np.zeros((3, 4)))
    for j in xrange(3):
    
        tmpLine = split(fileLines[j], ' ')
        m = 0
        
        # Assigning values into the transform matrix
        for k in xrange(len(tmpLine)):
            if (tmpLine[k] != ''):
                transformMatrix[j, m] = np.float(tmpLine[k])
                m += 1
                
    return transformMatrix

if __name__ == '__main__':
    
    # Read images and projection matrices
#    inDir = 'D:\\Dropbox\\Shared Folders\\PythonProject\\Code\\vtk_test\\data\\'
    inDir = '..\\..\\..\\vtk_test\\data\\'
    
    numImages = 7
    imgFileNamePrefix = inDir + 'ScanPos03 - Panorama001 - Image00'
    transformMatrixNamePrefix = inDir + 'P'
    
    # Creating the PointSet object and reading points from file        
    fileName = inDir + 'Hanover_scan3.xyz'
    
    pointSetList = []
    IOFactory.ReadXYZ(fileName, pointSetList)
    pointSet = pointSetList[0]   
    
    images = []
    transformMatrices = []
    for i in xrange(1, numImages + 1):
    
        # Read image
        img = imread(imgFileNamePrefix + str(i) + '.jpg')
        images.append(img)
    
        # Opening the transform matrix file
        matrixFileName = transformMatrixNamePrefix + str(i) + '.txt'
        transformMatrix = ReadProjectionMatrixFromFile(matrixFileName)
        transformMatrices.append(transformMatrix)

    colorProperty = ColorFactory.ProjectImages(pointSet, images, transformMatrices)
    Visualization.RenderPointSet(colorProperty, 'externRgb')      
    Visualization.Show()