import numpy as np

from FilterFactory import SlopeBasedMorphologicFilter
from IOFactory import IOFactory
from VisualizationVTK import VisualizationVTK

if __name__ == '__main__':
    
    pointSetList = []
    IOFactory.ReadXYZ('..\\Sample Data\\DU9_2.xyz', pointSetList)

#    filterFactory = FilterFactory()
    terrainSubSet = SlopeBasedMorphologicFilter(pointSetList[0], 1.0, 25 * np.pi / 180)
    
    print 'hi'
    
#    Visualization.RenderPointSet(terrainSubSet, 'color', color=(0.5, 0, 0))
    VisualizationVTK.RenderPointSet(terrainSubSet, 'rgb')
    VisualizationVTK.Show()
