from timeit import default_timer as timer

import numpy as np

from Cuda.cuda_functions import *


def compute_normals_by_tensor_cuda(pntSet, neighborsNumber, neighbors):
    """
    :param pntSet: the points set
    :param neighborsNumber: neighbors number for each point at the points set
    :param neighbors: neighbors for each point at the points set
    :return: normals for each point at the point set
    """
    pntSet = pntSet.T
    numPnts = pntSet.shape[1]
    nebNum = neighbors.shape[1]
    limit = 100000000
    partsNum = int(numPnts * nebNum / limit) + (numPnts * nebNum % limit > 0)
    print("start")
    stride = int(limit / (nebNum + 1))
    print("stride=", stride)
    start = timer()
    threads_per_block = 128
    # tensors_cpu = np.empty((3, 3 * numPnts))
    eigsVals_cpu = np.empty((3 * numPnts))
    eigVectors_cpu = np.empty((3 * numPnts))
    print("partsNum=", partsNum)
    for i in range(1, partsNum + 1):
        print(i)
        x = numPnts
        if i * stride > numPnts:
            # print(1)
            pntSet_device = cuda.to_device(pntSet[:, ((i - 1) * stride): numPnts])
            neighborsNumber_device = cuda.to_device(neighborsNumber[((i - 1) * stride):numPnts, :])
            neighbors_device = cuda.to_device(neighbors[((i - 1) * stride): numPnts, :])
            out = np.zeros([3, 3 * numPnts - 3 * ((i - 1) * stride)], dtype=float)
            out2 = np.zeros([3 * numPnts - 3 * ((i - 1) * stride)], dtype=float)
            tensors = cuda.device_array_like(out)
            eigsVals = cuda.device_array_like(out2)
            eigVectors = cuda.device_array_like(out2)
            x = int(numPnts - stride * (i - 1))


        else:
            pntSet_device = cuda.to_device(np.ascontiguousarray(pntSet[:, ((i - 1) * stride): (i * stride)]))
            neighborsNumber_device = cuda.to_device(neighborsNumber[((i - 1) * stride): (i * stride), :])
            neighbors_device = cuda.to_device(neighbors[((i - 1) * stride):(i * stride), :])
            out = np.zeros([3, 3 * stride], dtype=float)
            out2 = np.zeros([3 * stride], dtype=float)
            tensors = cuda.device_array_like(out)
            eigsVals = cuda.device_array_like(out2)
            eigVectors = cuda.device_array_like(out2)
            x = int(stride)

        blocks_per_grid = x
        cuda.synchronize()
        computeNormalByTensorGPU[blocks_per_grid, threads_per_block](pntSet_device, neighborsNumber_device,
                                                                     neighbors_device, tensors)

        cuda.synchronize()

        # ********************************************************************************************************************
        # start = timer()
        cuda.synchronize()
        computeEigenValuesGPU[8, threads_per_block](tensors, eigsVals)
        cuda.synchronize()
        # duration = timer() - start
        # print("gpu eigVals: ", duration)
        # ********************************************************************************************************************
        # start = timer()

        cuda.synchronize()
        computeEigenVectorGPU[8, threads_per_block](tensors, eigsVals, eigVectors)
        cuda.synchronize()

        # duration = timer() - start
        # print("gpu eigVecors: ", duration)

        if i * stride > numPnts:
            # tensors_cpu[3, 3 * numPnts - 3 * ((i - 1) * stride)] = tensors.copy_to_host()
            eigVectors_cpu[3 * ((i - 1) * stride):3 * numPnts] = eigVectors.copy_to_host()
        else:
            # tensors_cpu[3, 3 * numPnts - 3 * ((i - 1) * stride)] = tensors.copy_to_host()
            eigVectors_cpu[3 * ((i - 1) * stride):3 * (i * stride)] = eigVectors.copy_to_host()

    return eigVectors_cpu


def computeUmbrelaCurvatureCuda(pntSet, neighborsNumber, neighbors, normals):
    pntSet = pntSet.T
    numPnts = pntSet.shape[1]
    nebNum = neighbors.shape[1]
    limit = 100000000
    partsNum = int(numPnts * nebNum / limit) + (numPnts * nebNum % limit > 0)
    print("start")
    stride = int(limit / (nebNum + 1))
    print("stride=", stride)
    start = timer()
    threads_per_block = 128
    umbrelaCurvature = np.empty(numPnts)
    print("partsNum=", partsNum)
    for i in range(1, partsNum + 1):
        print(i)
        x = numPnts
        if i * stride > numPnts:
            # print(1)
            pntSet_device = cuda.to_device(pntSet[:, ((i - 1) * stride): numPnts])
            neighborsNumber_device = cuda.to_device(neighborsNumber[((i - 1) * stride):numPnts, :])
            neighbors_device = cuda.to_device(neighbors[((i - 1) * stride): numPnts, :])
            normals_device = cuda.to_device(normals[3 * (i - 1) * stride: 3 * numPnts])
            out = np.zeros([numPnts - ((i - 1) * stride)], dtype=float)
            out_device = cuda.to_device(out)
            x = int(numPnts - stride * (i - 1))


        else:
            pntSet_device = cuda.to_device(np.ascontiguousarray(pntSet[:, ((i - 1) * stride): (i * stride)]))
            neighborsNumber_device = cuda.to_device(neighborsNumber[((i - 1) * stride): (i * stride), :])
            neighbors_device = cuda.to_device(neighbors[((i - 1) * stride):(i * stride), :])
            normals_device = cuda.to_device(normals[3 * (i - 1) * stride: 3 * (i * stride)])
            out = np.zeros([(i * stride) - ((i - 1) * stride)], dtype=float)
            out_device = cuda.to_device(out)
            x = int(stride)

        blocks_per_grid = x
        cuda.synchronize()
        umbrelaCurvatureGPU[blocks_per_grid, threads_per_block](pntSet_device, normals_device, neighborsNumber_device,
                                                                neighbors_device, out_device)

        cuda.synchronize()
        if i * stride > numPnts:
            umbrelaCurvature[((i - 1) * stride):numPnts] = out_device.copy_to_host()
        else:
            umbrelaCurvature[((i - 1) * stride): (i * stride)] = out_device.copy_to_host()

    return umbrelaCurvature
