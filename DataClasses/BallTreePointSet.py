from functools import partial

from numpy import nonzero, logical_and, unique, zeros, array, hstack
from sklearn.neighbors import BallTree

from PointSet import PointSet


class BallTreePointSet(PointSet):

    def __init__(self, points, leafSize=40):
        super(PointSet, self).__init__()

        self.__ballTree = BallTree(points, leafSize)

        ballTreeData = self.__ballTree.get_arrays()  # Getting the data out of the ballTree object
        self.__ballTreePointIndexes = ballTreeData[1]  # Storing the internal indexes of the points
        self.__ballTreeNodes = ballTreeData[2]  # Storing the nodes data (mx4, ndarray)
        # self.__ballTreeNodeCenters = ballTreeData[3]  # Storing the centers of each node
        self.__numNodes = self.__ballTreeNodes.shape[0]  # Number of nodes in the tree
        self.__relations = []  # A list to define the tree hierarchy structure
        self.__ballTreeNodeHierarchy()  # Reconstructing the hierarchy of the tree
        self.__levels = zeros((self.__numNodes,), dtype=int)  # Array for storing the level of each node
        self.__ComputeNodeLevel(0)  # Computing the levels for all nodes
        self.__numPnts = points.shape[0]

    def __FindChildNodes(self, i):
        """
        Finds the child nodes of the i-th node in the tree
        :param i: The index of the node to search for its child nodes (int)
        :return: A dictionary with the indexes of the left and right child nodes, if they exist
        """
        childNodes = {'nodeId': i}
        if self.__ballTreeNodes['is_leaf'][i] == 0:  # Checking if node is not a leaf
            startIdx = self.__ballTreeNodes['idx_start'][i]  # The index of the first point in the node
            endIdx = self.__ballTreeNodes['idx_end'][i]  # The index of the last point in the node
            midIdx = int((startIdx + endIdx) / 2)  # The index of the middle point of the node

            # Finding the index of the left child of the node
            left = nonzero(logical_and(self.__ballTreeNodes['idx_start'][i + 1:] == startIdx,
                                       self.__ballTreeNodes['idx_end'][i + 1:] == midIdx))[0]

            # Finding the index of the right child of the node
            right = nonzero(logical_and(self.__ballTreeNodes['idx_start'][i + 1:] == midIdx,
                                        self.__ballTreeNodes['idx_end'][i + 1:] == endIdx))[0]

            if len(left) != 0:  # Checking if the left child node exists
                childNodes['leftChild'] = left[0] + i + 1  # Saving the index of left child node

            if len(right) != 0:  # Checking if the right child node exists
                childNodes['rightChild'] = right[0] + i + 1  # Saving the index of right child node

        return childNodes  # Return the dictionary with indexes of the child nodes

    def __SetParent(self, i):
        """
        Sets ths i-th node as the parent of its child nodes
        :param i: The index of the node to set as parent (int)
        """
        left = False
        right = False

        if 'leftChild' in self.__relations[i]:  # Checking if the node has a left child
            left = True
            # Setting i as the parent of the left child node
            if not ('parent' in self.__relations[self.__relations[i]['leftChild']]):
                self.__relations[self.__relations[i]['leftChild']]['parent'] = i

        if 'rightChild' in self.__relations[i]:  # Checking if the node has a right child
            right = True
            # Setting i as the parent of the right child node
            if not ('parent' in self.__relations[self.__relations[i]['rightChild']]):
                self.__relations[self.__relations[i]['rightChild']]['parent'] = i

        if left and right:
            self.__relations[self.__relations[i]['leftChild']]['sibling'] = self.__relations[i]['rightChild']
            self.__relations[self.__relations[i]['rightChild']]['sibling'] = self.__relations[i]['leftChild']

    def __ballTreeNodeHierarchy(self):
        """
        Reconstructing the tree hierarchy based on the internal indexes of the points in each node
        :return:
        """
        self.__relations = list(
            map(self.__FindChildNodes, range(self.__numNodes)))  # Finding the children for each node
        list(map(self.__SetParent, range(self.__numNodes)))  # Setting the parent for each node

    @property
    def numberOfNodes(self):
        """
        Retrieving the total number of nodes in the ball tree
        :return: The number of nodes in the ball tree
        """
        return self.__numNodes

    @property
    def ballTreeLeaves(self):
        """
        Retrieving the indexes of the nodes which are leaves
        :return: A list of indexes for all the nodes which are leaves
        """
        return nonzero(self.__ballTreeNodes['is_leaf'] == 1)[0]

    def getNodeRadius(self, index):
        """
        Retrieving the radius of a node given by its index
        :param index: The index of the node whose radius is requested (int)
        :return: The radius of the node (float)
        """
        return self.__ballTreeNodes['radius'][index]

    def getPointsOfNode(self, index):
        """
        Retrieving the indexes of the points in a given node
        :param index: The index of the node whose points are required (int)
        :return: The list of indexes of the points in the node (list)
        """
        # return map(self.__ballTreePointIndexes.__getitem__, range(self.__ballTreeNodes[index]['idx_start'],
        #                                                           self.__ballTreeNodes[index]['idx_end']))
        return self.__ballTreePointIndexes[list(range(self.__ballTreeNodes[index]['idx_start'],
                                                      self.__ballTreeNodes[index]['idx_end']))]

    def __getFirstNodeOfSize(self, index, radius, startingNode='root'):
        """
        Getting the first node along the one of the tree branches whose radius is larger than a given one
        :param index: The index of the node to begin the search by
        :param radius: The minimal required size of the node
        :return: The index of the first node with a radius larger than the given one
        """
        if startingNode == 'root':
            if self.__ballTreeNodes['radius'][index] < radius or self.__ballTreeNodes['is_leaf'][index] == 1:
                return index
            else:
                return hstack([self.__getFirstNodeOfSize(self.getLeftChildOfNode(index), radius, 'root'),
                               self.__getFirstNodeOfSize(self.getRightChildOfNode(index), radius, 'root')])

        elif startingNode == 'leaves':
            if self.__ballTreeNodes['radius'][index] < radius:

                while self.__ballTreeNodes['radius'][self.__relations[index]['parent']] < radius:
                    index = self.__relations[index]['parent']

            return index
        else:
            return None

    def getSmallestNodesOfSize(self, radius, startingNode='root'):
        """
        Getting a list of the smallest nodes whose radii are larger than a given radius
        :param radius: The minimal required size of the nodes
        :return:
        """
        if startingNode == 'root':
            return self.__getFirstNodeOfSize(0, radius=radius, startingNode='root')
        elif startingNode == 'leaves':
            leavesIndexes = self.ballTreeLeaves
            return unique(list(map(partial(self.__getFirstNodeOfSize, radius=radius), leavesIndexes)))
        else:
            return None

    def __ComputeNodeLevel(self, index):
        """
        Computing the level of a node in the tree defined by its index. If the node has child nodes the computation is
        done for them as well. The levels of the node and its children are updated in the internal array of levels
        :param index: The index of the node
        """
        if not ('parent' in self.__relations[index]):  # Checking if the node is the root
            self.__levels[index] = 0
        else:
            self.__levels[index] = self.__levels[self.__relations[index]['parent']] + 1

        if 'leftChild' in self.__relations[index]:  # Checking if the node has a left child
            self.__ComputeNodeLevel(self.__relations[index]['leftChild'])
        if 'rightChild' in self.__relations[index]:  # Checking if the node has a right child
            self.__ComputeNodeLevel(self.__relations[index]['rightChild'])

    def getNodeLevel(self, index):
        """
        Retrieving the level of a node defined by its index
        :param index: The index of the node whose level is required
        :return: The level of the node
        """
        return self.__levels[index]

    @property
    def maxLevel(self):
        """
        Retrieving the maximum level of the tree
        :return: Returns the maximum level of the tree
        """
        return self.__levels.max()

    def query(self, pnts, k):
        """
        Query the ball tree for the k nearest neighbors of a given set of points
        :param pnts: The query points (ndarray, nx3)
        :param k: The number of neighbors to find for the point
        :return: The indexes for the neighbors of the points
        """
        distances, indexes = self.__ballTree.query(pnts, k=k)
        return indexes

    def queryRadius(self, pnts, radius):
        """
        Query the ball tree to find the neighbors of a given set of point inside a given radius
        :param pnts: The query points (ndarray, nx3)
        :param radius: The query radius
        :return: The indexes for the neighbors of the points
        """
        if isinstance(pnts, list):
            pnts = array(pnts)

        if pnts.ndim == 1:
            indexes = self.__ballTree.query_radius(pnts.reshape((1, -1)), radius)

            if indexes.dtype == object:
                indexes = indexes[0]

        else:
            indexes = self.__ballTree.query_radius(pnts, radius)

        return indexes

    def getLeftChildOfNode(self, index):
        """
        Retrieving the index of the left child node of a given node
        :param index: The index of the node whose left child is required
        :return: The index of the left child node or None, if the node does not have a left child
        """
        if 'leftChild' in self.__relations[index]:
            return self.__relations[index]['leftChild']
        else:
            return None

    def getRightChildOfNode(self, index):
        """
        Retrieving the index of the right child node of a given node
        :param index: The index of the node whose right child is required
        :return: The index of the right child node or None, if the node does not have a right child
        """
        if 'rightChild' in self.__relations[index]:
            return self.__relations[index]['rightChild']
        else:
            return None

    def getParentOfNode(self, index):
        """
        Retrieving the index of the parent node of a given node
        :param index: The index of the node whose parent is required
        :return: The index of the parent node or None, if the node does not have a parent (node is the root)
        """
        if 'parent' in self.__relations[index]:
            return self.__relations[index]['parent']
        else:
            return None

    def getSiblingOfNode(self, index):
        """
        Retrieving the index of the sibling node of a given node
        :param index: The index of the node whose sibling is required
        :return: The index of the sibling node or None, if the node does not have a sibling
        """
        if 'sibling' in self.__relations[index]:
            return self.__relations[index]['sibling']
        else:
            return None

    def ToNumpy(self):
        return self.__ballTree.get_arrays()[0]

    def ToPolyData(self):
        from VisualizationUtils import MakeVTKPointsMesh
        vtkPolyData = MakeVTKPointsMesh(self.ToNumpy())
        return vtkPolyData

    @property
    def Size(self):
        return self.__numPnts

    @property
    def X(self):
        return self.ToNumpy()[:, 0]

    @property
    def Y(self):
        return self.ToNumpy()[:, 1]

    @property
    def Z(self):
        return self.ToNumpy()[:, 2]

    def save(self, path_or_buf, **kwargs):
        # TODO: IMPLEMENT save to json file
        pass


if __name__ == '__main__':
    from numpy.random import random

    points = random((10000, 3))
    ballTree = BallTreePointSet(points)
    print(ballTree.numberOfNodes)
    print(ballTree.ballTreeLeaves)
    print(ballTree.getNodeRadius(0))
    print(ballTree.getPointsOfNode(240))
    print(ballTree.maxLevel)
    print(ballTree.getNodeLevel(0))
    print(ballTree.getNodeLevel(50))

    print(ballTree.query([[0.5, 0.5, 0.5], [0.75, 0.75, 0.75]], 7))
    print(ballTree.queryRadius([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], 0.1))

    print(ballTree.getLeftChildOfNode(0))
    print(ballTree.getRightChildOfNode(0))
    print(ballTree.getSiblingOfNode(1))
    print(ballTree.getSiblingOfNode(2))
    print(ballTree.getParentOfNode(2))
    print(ballTree.getParentOfNode(0))
    print(ballTree.getLeftChildOfNode(240))
    print(ballTree.getRightChildOfNode(240))
    print(ballTree.getSiblingOfNode(0))
