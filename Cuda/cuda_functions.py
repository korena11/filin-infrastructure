import math

from numba import cuda, double


# @cuda.jit(['float64(int64,float64[:,:],float64[:,:])'],device=True)
@cuda.jit(device=True)
def diff(i, tmp, pnts, ind, points_before_rot):
    for j in range(tmp.shape[0]):
        p = ind[i, 1 + j]
        # k = i + j
        tmp[j:, 0] = pnts[p, 0] - pnts[i, 0]
        tmp[j:, 1] = pnts[p, 1] - pnts[i, 1]
        tmp[j:, 2] = pnts[p, 2] - pnts[i, 2]
        points_before_rot[j, 0 + i * 3] = tmp[j, 0]
        points_before_rot[j, 1 + i * 3] = tmp[j, 1]
        points_before_rot[j, 2 + i * 3] = tmp[j, 2]
    return tmp


@cuda.jit(device=True)
def product(tmp, out, k):
    for i in range(tmp.shape[1]):
        for j in range(tmp.shape[1]):
            sum = 0
            for l in range(tmp.shape[0]):
                sum += tmp[l, i] * tmp[l, j]
                # print(sum)
            # out[3 * k + i, 3 * k + j] = sum
            out[i, 3 * k + j] = sum


@cuda.jit(device=True)
def trace_3(mat):
    sum = 0
    sum += mat[0, 0]
    sum += mat[1, 1]
    sum += mat[2, 2]
    return sum


@cuda.jit(device=True)
def getMat_3(input, i, A):
    A[0, 0] = input[0, i * 3]
    A[0, 1] = input[0, i * 3 + 1]
    A[0, 2] = input[0, i * 3 + 2]
    A[1, 0] = input[1, i * 3 + 0]
    A[1, 1] = input[1, i * 3 + 1]
    A[1, 2] = input[1, i * 3 + 2]
    A[2, 0] = input[2, i * 3 + 0]
    A[2, 1] = input[2, i * 3 + 1]
    A[2, 2] = input[2, i * 3 + 2]


@cuda.jit(device=True)
def copy_mat_3(A, out):
    mulScalar_3(A, out, 1)


@cuda.jit(device=True)
def mulScalar_3(A, out, x):
    out[0, 0] = x * A[0, 0]
    out[0, 1] = x * A[0, 1]
    out[0, 2] = x * A[0, 2]
    out[1, 0] = x * A[1, 0]
    out[1, 1] = x * A[1, 1]
    out[1, 2] = x * A[1, 2]
    out[2, 0] = x * A[2, 0]
    out[2, 1] = x * A[2, 1]
    out[2, 2] = x * A[2, 2]


@cuda.jit(device=True)
def add_scalar_to_diagonal_3(A, x):
    A[0, 0] += x
    A[1, 1] += x
    A[2, 2] += x


@cuda.jit(device=True)
def det_2(A):
    res = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    return res


@cuda.jit(device=True)
def det_3(A):
    res = A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) + A[
        0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    return res


@cuda.jit(device=True)
def print_3(A):
    print(A[0, 0], A[0, 1], A[0, 2], "\n",
          A[1, 0], A[1, 1], A[1, 2], "\n",
          A[2, 0], A[2, 1], A[2, 2])


@cuda.jit(device=True)
def print_vector_3(v):
    print(v[0], v[1], v[2])


@cuda.jit(device=True)
def squared_trace_3(A):
    return (A[0, 0] * A[0, 0] + A[0, 1] * A[1, 0] + A[0, 2] * A[2, 0]) + (
            A[1, 0] * A[0, 1] + A[1, 1] * A[1, 1] + A[1, 2] * A[2, 1]) + (
                   A[2, 0] * A[0, 2] + A[2, 1] * A[1, 2] + A[2, 2] * A[2, 2])


@cuda.jit(device=True)
def norm_1x3(v):
    sum = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
    sum = sum ** 0.5
    return sum


@cuda.jit(device=True)
def cross_product_1x3(v1, v2, v):
    v[0] = v1[1] * v2[2] - v1[2] * v2[1]
    v[1] = -v1[0] * v2[2] + v1[2] * v2[0]
    v[2] = v1[0] * v2[1] - v1[1] * v2[0]


@cuda.jit(device=True)
def normalized_1x3(v):
    norm = norm_1x3(v)
    v[0] = v[0] / norm
    v[1] = v[1] / norm
    v[2] = v[2] / norm


@cuda.jit(device=True)
def getVector_1x3(aignVectors, i, v):
    v[0] = aignVectors[i * 3]
    v[1] = aignVectors[i * 3 + 1]
    v[2] = aignVectors[i * 3 + 2]


@cuda.jit(device=True)
def rotateVector_1x3(R, v_in, v_out):
    v_out[0] = v_in[0] * R[0, 0] + v_in[1] * R[0, 1] + v_in[2] * R[0, 2]
    v_out[1] = v_in[0] * R[1, 0] + v_in[1] * R[1, 1] + v_in[2] * R[1, 2]
    v_out[2] = v_in[0] * R[2, 0] + v_in[1] * R[2, 1] + v_in[2] * R[2, 2]


@cuda.jit(device=True)
def mat_x_Vector_1x3(N, v, v_out):
    rotateVector_1x3(N, v, v_out)


@cuda.jit(device=True)
def A33_inverse(A33, A33_inv):
    det = det_3(A33)
    if det == 0:
        print("inverse error 0 det")
    A33_inv[0][0] = (A33[1][1] * A33[2][2] - A33[1][2] * A33[2][1]) / det
    A33_inv[0][1] = -(A33[0][1] * A33[2][2] - A33[0][2] * A33[2][1]) / det
    A33_inv[0][2] = (A33[0][1] * A33[1][2] - A33[0][2] * A33[1][1]) / det
    A33_inv[1][0] = A33_inv[0][1]
    A33_inv[1][1] = (A33[0][0] * A33[2][2] - A33[0][2] * A33[2][0]) / det
    A33_inv[1][2] = -(A33[0][0] * A33[1][2] - A33[0][2] * A33[1][0]) / det
    A33_inv[2][0] = A33_inv[0][2]
    A33_inv[2][1] = A33_inv[1][2]
    A33_inv[2][2] = (A33[0][0] * A33[1][1] - A33[0][1] * A33[1][0]) / det


@cuda.jit(device=True)
def scalarProduct_1x3(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    v1[0] = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


# @cuda.jit('void(float64[:,:], int64[:,:], float64[:,:], float64[:,:])')
@cuda.jit()
def computeNormalByTensorGPU(pnts, neighborsNumber, neighbors, tensors):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    # grid_size = cuda.gridDim.x
    # start = tx + ty * block_size
    start = tx
    # stride = block_size * grid_size
    stride = block_size
    # tmp = cuda.local.array(shape=(10, 3), dtype=double)
    sharedTmp = cuda.shared.array(shape=(3, 128), dtype=double)
    point = cuda.shared.array(shape=(3, 1), dtype=double)
    tensor = cuda.shared.array(shape=(3, 3 * 128), dtype=double)
    # tensor = cuda.shared.array(shape=(3, 3), dtype=double)
    neighborsNum = neighborsNumber[ty, 0]
    # import numpy
    # tmp = cuda.local.array(numpy.empty([10, 3], dtype=float))
    if (tx == 0):
        point[0, 0] = pnts[0, ty]
        point[1, 0] = pnts[1, ty]
        point[2, 0] = pnts[2, ty]
    for k in range(0, neighborsNum, stride):
        end = ((neighborsNum >= (k + stride)) * (k + stride)) + (
                (neighborsNum < (k + stride)) * neighborsNum)

        for i in range(start + k, end, stride):
            j = i - k

            sharedTmp[0, j] = neighbors[3 * ty, i]
            sharedTmp[1, j] = neighbors[3 * ty + 1, i]
            sharedTmp[2, j] = neighbors[3 * ty + 2, i]
        cuda.syncthreads()
        for i in range(start + k, end, stride):
            j = i - k

            sharedTmp[0, j] -= point[0, 0]
            sharedTmp[1, j] -= point[1, 0]
            sharedTmp[2, j] -= point[2, 0]
            # if ty ==2 and tx==9:
            #     print(sharedTmp[0, j])
            #     print(sharedTmp[1, j])
            #     print(sharedTmp[2, j])
            # points_before_rot[0,ty*11+j]=sharedTmp
        cuda.syncthreads()
        # if (neighborsNum < (end )):
        #     print("yes",k,tx)
        #     for i in range(start, end, stride):
        for i in range(start, 128, stride):
            j = i
            tensor[0, j * 3] = 0
            tensor[0, j * 3 + 1] = 0
            tensor[0, j * 3 + 2] = 0
            tensor[1, j * 3] = 0
            tensor[1, j * 3 + 1] = 0
            tensor[1, j * 3 + 2] = 0
            tensor[2, j * 3] = 0
            tensor[2, j * 3 + 1] = 0
            tensor[2, j * 3 + 2] = 0

            tensor[0, tx * 3] = 0
            tensor[0, tx * 3 + 1] = 0
            tensor[0, tx * 3 + 2] = 0
            tensor[1, tx * 3] = 0
            tensor[1, tx * 3 + 1] = 0
            tensor[1, tx * 3 + 2] = 0
            tensor[2, tx * 3] = 0
            tensor[2, tx * 3 + 1] = 0
            tensor[2, tx * 3 + 2] = 0
        cuda.syncthreads()

        for i in range(start + k, end, stride):
            j = i - k
            tensor[0, j * 3] = sharedTmp[0, j] * sharedTmp[0, j]
            tensor[0, j * 3 + 1] = sharedTmp[0, j] * sharedTmp[1, j]
            tensor[0, j * 3 + 2] = sharedTmp[0, j] * sharedTmp[2, j]
            tensor[1, j * 3] = sharedTmp[1, j] * sharedTmp[0, j]
            tensor[1, j * 3 + 1] = sharedTmp[1, j] * sharedTmp[1, j]
            tensor[1, j * 3 + 2] = sharedTmp[1, j] * sharedTmp[2, j]
            tensor[2, j * 3] = sharedTmp[2, j] * sharedTmp[0, j]
            tensor[2, j * 3 + 1] = sharedTmp[2, j] * sharedTmp[1, j]
            tensor[2, j * 3 + 2] = sharedTmp[2, j] * sharedTmp[2, j]

        cuda.syncthreads()
        # if tx==0:
        #
        #     print(1)
        len = int(stride / 2)

        while len >= 1:
            for j in range(tx, len, stride):
                tx_ = j * 3
                m = j + len
                m *= 3
                tensor[0, tx_] += tensor[0, m]
                tensor[0, tx_ + 1] += tensor[0, m + 1]
                tensor[0, tx_ + 2] += tensor[0, m + 2]
                tensor[1, tx_] += tensor[1, m]
                tensor[1, tx_ + 1] += tensor[1, m + 1]
                tensor[1, tx_ + 2] += tensor[1, m + 2]
                tensor[2, tx_] += tensor[2, m]
                tensor[2, tx_ + 1] += tensor[2, m + 1]
                tensor[2, tx_ + 2] += tensor[2, m + 2]
            cuda.syncthreads()
            len = int(len / 2)

        if tx == 0:
            tensors[0, 3 * ty] += tensor[0, 0]
            tensors[0, 3 * ty + 1] += tensor[0, 1]
            tensors[0, 3 * ty + 2] += tensor[0, 2]
            tensors[1, 3 * ty] += tensor[1, 0]
            tensors[1, 3 * ty + 1] += tensor[1, 1]
            tensors[1, 3 * ty + 2] += tensor[1, 2]
            tensors[2, 3 * ty] += tensor[2, 0]
            tensors[2, 3 * ty + 1] += tensor[2, 1]
            tensors[2, 3 * ty + 2] += tensor[2, 2]

        start += stride

    # cuda.atomic.add(tensors, (0, 0), a00)
    # cuda.atomic.add(tensors,(0,1),a01)
    # cuda.atomic.add(tensors,(0,2),a02)
    # cuda.atomic.add(tensors,(1,0),a10)
    # cuda.atomic.add(tensors,(1,1),a11)
    # cuda.atomic.add(tensors,(1,2),a12)
    # cuda.atomic.add(tensors,(2,0),a20)
    # cuda.atomic.add(tensors,(2,1),a21)
    # cuda.atomic.add(tensors,(2,2),a22)


# for i in range(start, pnts.shape[1], stride):
#     tmp = diff(i, tmp, pnts, ind, points_before_rot)
#     product(tmp, tensors, i)
#


@cuda.jit()
def computeEigenValuesGPU(input, out):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    grid_size = cuda.gridDim.x
    stride = block_size * grid_size
    start = tx + ty * block_size
    size = int(input.shape[1] / 3)
    # size = math.floor(size+0.5)
    # print(size)
    A = cuda.local.array(shape=(3, 3), dtype=double)
    tmp = cuda.local.array(shape=(3, 3), dtype=double)

    for i in range(start, size, stride):
        getMat_3(input, i, A)
        q = trace_3(A) / 3

        mulScalar_3(A, tmp, -1)

        add_scalar_to_diagonal_3(tmp, q)

        p = (squared_trace_3(tmp) / 6) ** 0.5

        add_scalar_to_diagonal_3(A, -q)
        mulScalar_3(A, tmp, 1 / p)
        det = det_3(tmp)
        phi = math.acos(det / 2) / 3
        eig1 = q + 2 * p * math.cos(phi)
        eig3 = q + 2 * p * math.cos(phi + 2 * math.pi / 3.0)
        eig2 = 3 * q - eig1 - eig3
        j = i * 3
        if eig1 < eig2 and eig1 < eig3:
            out[j] = eig1
            if eig2 < eig3:
                out[j + 2] = eig3
                out[j + 1] = eig2

            else:
                out[j + 2] = eig2
                out[j + 1] = eig3

        elif eig2 < eig1 and eig2 < eig3:
            out[j] = eig2
            if eig1 < eig3:
                out[j + 2] = eig3
                out[j + 1] = eig1

            else:
                out[j + 2] = eig1
                out[j + 1] = eig3

        else:
            out[j] = eig3
            if eig1 < eig2:
                out[j + 2] = eig2
                out[j + 1] = eig1

            else:
                out[j + 2] = eig1
                out[j + 1] = eig2

    # out[j + 2] = eig2
    # out[j + 1] = eig1
    # out[j ] = eig3


@cuda.jit()
def computeEigenVectorGPU(input1, eigVals, out):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    grid_size = cuda.gridDim.x
    stride = block_size * grid_size
    start = tx + ty * block_size
    size = int(input1.shape[1] / 3)

    A = cuda.local.array(shape=(3, 3), dtype=double)
    v = cuda.local.array(shape=3, dtype=double)

    for i in range(start, size, stride):
        getMat_3(input1, i, A)
        add_scalar_to_diagonal_3(A, -1 * eigVals[i * 3])
        cross_product_1x3(A[0], A[1], v)  # we didn't check if the rows is not Linear independent

        normalized_1x3(v)
        # if i == 0 or i == 1 or i == 2:
        #     print(v[0], v[1], v[2])
        j = i * 3
        out[j] = v[0]
        out[j + 1] = v[1]
        out[j + 2] = v[2]


@cuda.jit()
def computeRotationOfNormalToZaxisGPU(aigVecotors, out):
    """
    Computing the rotation matrix for rotating a surface so that its normal will be in the direction of the Z-axis
    :param n: normal vector (1-D ndarray, length = 3)
    :return: 3-D rotation matrix (ndarray, 3x3)
    """
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    grid_size = cuda.gridDim.x
    stride = block_size * grid_size
    start = tx + ty * block_size
    size = int(out.shape[1] / 3)
    n = cuda.local.array(shape=3, dtype=double)
    # n2 = cuda.local.array(shape=3, dtype=float64)
    # rotMat = cuda.local.array(shape=(3, 3), dtype=float64)

    for i in range(start, size, stride):
        getVector_1x3(aigVecotors, i, n)
        out[0, 0 + i * 3] = 1 - n[0] ** 2 / (n[2] + 1)
        out[0, 1 + i * 3] = -n[0] * n[1] / (n[2] + 1)
        out[0, 2 + i * 3] = -n[0]
        out[1, 0 + i * 3] = -n[0] * n[1] / (n[2] + 1)
        out[1, 1 + i * 3] = 1 - n[1] ** 2 / (n[2] + 1)
        out[1, 2 + i * 3] = -n[1]
        out[2, 0 + i * 3] = n[0]
        out[2, 1 + i * 3] = n[1]
        out[2, 2 + i * 3] = n[2]
        # rotMat[0, 0] = 1 - n[0] ** 2 / (n[2] + 1)
        # rotMat[0, 1] = -n[0] * n[1] / (n[2] + 1)
        # rotMat[0, 2] = -n[0]
        # rotMat[1, 0] = -n[0] * n[1] / (n[2] + 1)
        # rotMat[1, 1] = 1 - n[1] ** 2 / (n[2] + 1)
        # rotMat[1, 2] = -n[1]
        # rotMat[2, 0] = n[0]
        # rotMat[2, 1] = n[1]
        # rotMat[2, 2] = n[2]
        # getMat_3(out, i, rotMat)
        # print_vector_3(n)
        #
        # rotateVector_1x3(rotMat,n,n2)
        # print_vector_3(n2)


@cuda.jit()
def rotatePointGPU(points_before_rot, rotMats):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    grid_size = cuda.gridDim.x
    stride = block_size * grid_size
    start = tx + ty * block_size
    size = int(rotMats.shape[1] / 3)
    v_in = cuda.local.array(shape=3, dtype=double)
    v_out = cuda.local.array(shape=3, dtype=double)
    rotMat = cuda.local.array(shape=(3, 3), dtype=double)

    for i in range(start, size, stride):
        getMat_3(rotMats, i, rotMat)
        for j in range(10):
            v_in[0] = points_before_rot[j, i * 3]
            v_in[1] = points_before_rot[j, i * 3 + 1]
            v_in[2] = points_before_rot[j, i * 3 + 2]
            rotateVector_1x3(rotMat, v_in, v_out)

            points_before_rot[j, i * 3] = v_out[0]
            points_before_rot[j, i * 3 + 1] = v_out[1]
            points_before_rot[j, i * 3 + 2] = v_out[2]


@cuda.jit()
def calcBiquadraticVals(points, out, out2):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    grid_size = cuda.gridDim.x
    stride = block_size * grid_size
    start = tx + ty * block_size
    size = int(points.shape[1] / 3)
    v_out = cuda.local.array(shape=3, dtype=double)
    u = cuda.local.array(shape=3, dtype=double)
    N = cuda.local.array(shape=(3, 3), dtype=double)
    N_inv = cuda.local.array(shape=(3, 3), dtype=double)
    for i in range(start, size, stride):

        mulScalar_3(N, N, 0)
        u[0] = 0
        u[1] = 0
        u[2] = 0
        for j in range(10):
            N[0, 0] += points[j, i * 3] ** 4
            N[0, 1] += (points[j, i * 3] ** 3) * points[j, i * 3 + 1]
            N[0, 2] += (points[j, i * 3] ** 2) * (points[j, i * 3 + 1] ** 2)
            N[1, 2] += (points[j, i * 3]) * (points[j, i * 3 + 1] ** 3)
            N[2, 2] += points[j, i * 3 + 1] ** 4
            u[0] += (points[j, i * 3] ** 2) * points[j, i * 3 + 2]
            u[1] += (points[j, i * 3]) * (points[j, i * 3 + 1]) * points[j, i * 3 + 2]
            u[2] += (points[j, i * 3 + 1] ** 2) * points[j, i * 3 + 2]

        N[1, 1] = N[0, 2]
        N[1, 0] = N[0, 1]
        N[2, 0] = N[0, 2]
        N[2, 1] = N[1, 2]
        A = N
        out[0, 0 + i * 3] = A[0, 0]
        out[0, 1 + i * 3] = A[0, 1]
        out[0, 2 + i * 3] = A[0, 2]
        out[1, 0 + i * 3] = A[1, 0]
        out[1, 1 + i * 3] = A[1, 1]
        out[1, 2 + i * 3] = A[1, 2]
        out[2, 0 + i * 3] = A[2, 0]
        out[2, 1 + i * 3] = A[2, 1]
        out[2, 2 + i * 3] = A[2, 2]

        A33_inverse(N, N_inv)
        mat_x_Vector_1x3(N_inv, u, v_out)

        out2[0, 0 + i * 3] = v_out[0]
        out2[1, 1 + i * 3] = v_out[1]
        out2[2, 2 + i * 3] = v_out[2]
        # print_vector_3(v_out)


@cuda.jit()
def umbrelaCurvatureGPU(normals, vectors, out):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    block_size = cuda.blockDim.x

    grid_size = cuda.gridDim.x
    stride = block_size * grid_size
    start = tx + ty * block_size
    size = int(vectors.shape[1] / 3)
    v = cuda.local.array(shape=3, dtype=double)
    normal = cuda.local.array(shape=3, dtype=double)

    for i in range(start, size, stride):
        normal[0] = normals[0 + i * 3]
        normal[1] = normals[1 + i * 3]
        normal[2] = normals[2 + i * 3]
        norm = norm_1x3(normal)
        out[i] = 0

        for j in range(vectors.shape[0]):
            v[0] = vectors[j, 0 + i * 3]
            v[1] = vectors[j, 1 + i * 3]
            v[2] = vectors[j, 2 + i * 3]
            norm_v = norm_1x3(v)
            out[i] += (scalarProduct_1x3(normal, v)) / (norm_v * norm)


@cuda.jit()
def tensorRank_1(vectors, out):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    block_size = cuda.blockDim.x

    grid_size = cuda.gridDim.x
    stride = block_size * grid_size
    start = tx + ty * block_size
    size = int(vectors.shape[1] / 3)
    v = cuda.local.array(shape=3, dtype=double)
    v[0] = 0
    v[1] = 0
    v[2] = 0
    for i in range(start, size, stride):
        for j in range(vectors.shape[0]):
            v[0] += vectors[j, 0 + i * 3]
            v[1] += vectors[j, 1 + i * 3]
            v[2] += vectors[j, 2 + i * 3]
        out[0, 0 + i * 3] = v[0]
        out[1, 1 + i * 3] = v[1]
        out[2, 2 + i * 3] = v[2]
