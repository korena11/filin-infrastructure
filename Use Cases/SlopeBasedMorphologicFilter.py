import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
from numba import autojit
from IOFactory import IOFactory
from FilterFactory import SlopeBasedMorphologicFilter
from Visualization import Visualization 

if __name__ == '__main__':
    
    pointSetList = []
    IOFactory.ReadXYZ('..\\Sample Data\\DU9_2.xyz', pointSetList)

#    filterFactory = FilterFactory()
    terrainSubSet = SlopeBasedMorphologicFilter(pointSetList[0], 1.0, 25 * np.pi / 180)
    
    print 'hi'
    
#    Visualization.RenderPointSet(terrainSubSet, 'color', color=(0.5, 0, 0))
    Visualization.RenderPointSet(terrainSubSet, 'rgb')
    Visualization.Show()
      