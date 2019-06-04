from numpy import eye


class TensorSet(object):

    def __init__(self, tensors=None):

        if tensors is None:
            self.__covMat = eye(3)
            self.__numTensors = 0
            self.__numPoints = 0
            self.__tensors = None
        else:
            # TODO: complete implementation
            pass

    def distanceFromPoint(self, pnt):
        # TODO: complete implementation
        return None
