from IOFactory import IOFactory
from KdTreePointSet import KdTreePointSet
from NeighborsFactory import NeighborsFactory

if __name__ == '__main__':
    path = '../../test_data/'
    filename = 'test_pts'
    pntSet = IOFactory.ReadPts(path + filename + '.pts')
    print(pntSet.Size)

    pntSet = KdTreePointSet(pntSet.ToNumpy())

    neighbors = NeighborsFactory.kdtreePointSet_rnn(pntSet, 0.05)
    neighbors.ToCUDA()

