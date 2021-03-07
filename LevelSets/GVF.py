import numpy as np


"""
Codes for Gradient vector flow computation

Translated from Chenyang Xu and Jerry L. Prince, 9/9/1999
`Xu GVF webpage <http://iacl.ece.jhu.edu/projects/gvf>`_.  

"""

def boundMirrorEnsure(A):
    """
    Ensure mirror boundary condition

    The number of rows and columns of A must be greater than 2

    for example (X means value that is not of interest)

    A = [
        X  X  X  X  X   X
        X  1  2  3  11  X
        X  4  5  6  12  X
        X  7  8  9  13  X
        X  X  X  X  X   X
        ]

    B = BoundMirrorEnsure(A) will yield

        5  4  5  6  12  6
        2  1  2  3  11  3
        5  4  5  6  12  6
        8  7  8  9  13  9
        5  4  5  6  12  6


    :param A: the matrix to ensure its boundaries

    :type A: np.array

    :return:  a re-adjusted matrix

    :rtype: np.array


    """

    m, n = A.shape[:2]

    if (m < 3 or n < 3):
        raise ValueError('either the number of rows or columns is smaller than 3')


    # yi = np.arange(1, m-1)
    # xi = np.arange(1, n-1)

    A[[0,m-1], [0, n-1]] = A[[2, m - 3], [2, n - 3]] # mirror corners
    A[[0, m-1], 1:-2] = A[[2, m - 3], 1:-2] # mirror left and right boundary
    A[1:-2, [0, n-1]] = A[1:-2, [2, n - 3]] # mirror  top and bottom boundary

    return A

def boundMirrorExpand(A):
    """
     Expand the matrix using mirror boundary condition

     for example

     A = [
         1  2  3  11
         4  5  6  12
         7  8  9  13
         ]

     B = BoundMirrorExpand(A) will yield

         5  4  5  6  12  6
         2  1  2  3  11  3
         5  4  5  6  12  6
         8  7  8  9  13  9
         5  4  5  6  12  6


    :param A: the matrix to ensure its boundaries

    :type A: np.array

    :return:  a re-adjusted matrix

    :rtype: np.array
    """
    m, n = A.shape[:2]
    # yi = np.arange(1, m+1)
    # xi = np.arange(1, n+1)

    B = np.zeros((m + 2, n + 2))
    B[1:-1, 1:-1] = A

    B[[0, m + 1], [0, n+1]] = B[[2, m-1], [2, n-1]] # mirror top-left, right-bottom corners
    B[[m+1, 0], [0, n+1]] = B[[2, m-1], [2,n-1]]
    B[[0, m + 1], 1:-1] = B[[2, m-1], 1:-1] # mirror left and right boundary
    B[ 1:-1, [0, n + 1]] = B[1:-1, [2, n-1]] # mirror  top and bottom boundary

    return B

def boundMirrorShrink(A):
    """
     Shrink the matrix to remove the padded mirror boundaries

     for example

     A = [
         5  4  5  6  12  6
         2  1  2  3  11  3
         5  4  5  6  12  6
         8  7  8  9  13  9
         5  4  5  6  12  6
         ]

     B = BoundMirrorShrink(A) will yield

         1  2  3  11
         4  5  6  12
         7  8  9  13

    :param A: the matrix to ensure its boundaries

    :type A: np.array

    :return:  a re-adjusted matrix

    :rtype: np.array
    """

    return A[1:-1, 1:-1]

def GVF(f, mu, iterations, ksize=3, sigma=1.,
                          resolution=1., blur_window=(0,0), **kwargs):
    """
    GVF Compute gradient vector flow of an edge map f.

    :param f: the edge map to compute the GVF for
    :param mu: regularization coefficient
    :param iterations: number of iterations that will be computed.
    :param ksize: size of the differentiation window
    :param resolution: kernel resolution
    :param sigma: sigma for gaussian blurring. Default: 1. If sigma=0 no smoothing is carried out
    :param blur_window: tuple of window size for blurring


    :return: the GVF [u,v]

    :rtype: np.array nxmx2
    """
    import MyTools as mt
    from scipy.ndimage.filters import laplace as del2
    from tqdm import tqdm
    from MyTools import scale_values
    # normalize f to the range [0,1]
    f = scale_values(f)
    m, n = f.shape[:2]

    f = boundMirrorExpand(f) # Take care of boundary condition
    [fx, fy] = mt.imageDerivatives_4connected(f, 1, ksize=ksize, sigma=sigma, resolution=resolution,
                                              blur_window=blur_window) # Calculate the gradient of the edge map

    # Initialize  GVF to the gradient
    u = fx
    v = fy

    SqrMagf = fx * fx + fy * fy # Squared magnitude of the gradient field

    # Iteratively solve for the GVF u, v
    for i in tqdm(np.arange(0,iterations),desc='Creating GVF'):
        u = boundMirrorEnsure(u)
        v = boundMirrorEnsure(v)
        u +=  mu  * del2(u) - SqrMagf * (u - fx)
        v +=  mu  * del2(v) - SqrMagf * (v - fy)

    u = boundMirrorShrink(u)
    v = boundMirrorShrink(v)

    return np.stack((u, v), axis=2)
