import numpy as np
from tqdm import tqdm


def adaptive_smoothing(panorama, phenomena_size, **kwargs):
    r"""
    Adaptive smoothing of range image according to another (self) panorama image

    .. note::
       The function is implemented similar to other smoothing functions (e.g., `cv2.GaussianBlur`; `cv2.Blur`), so
       different kinds of smoothing functions will be applicable by sending (without `if`).
       This means that the function will return computed `sigma` and `ksize` according to
       the average - and they will be returned in the variable.

     An adaptive smoothing is implemented as a family of convolution kernels characterized by
     different :math:`\sigma` values where:

    .. math::
         d(\rho )=\frac{D}{\rho \Delta }

    with :math: `D`, the object-space size of the window, and :math: `\Delta` the angular sampling resolution
    and :math: `\rho` is the measured range.

    :cite:`Arav2013`

    .. todo::
       Add optionality for adaptive smoothing for other properties panoramas, where adaptation is according to
       range information

    :param phenomena_size: the minimal size of the phenomena that we don't want to smooth (:math: `D` above)
     default: 0.2 m

    :param panorama: the panorama (property) to smooth

    :type phenomena_size: float
    :type panorama: Properties.Panoramas.PanoramaProperty.PanoramaProperty


    :return:
       - smoothed image
       - mean sigma size
       - mean kernel size

    :rtype: tuple

    """
    import cv2

    if phenomena_size == 0:
        phenomena_size = 0.2  # in meters
    filtered_image = np.zeros(panorama.Size)
    sigmas = []
    ksizes = []
    scan_resolution = np.mean([panorama.azimuth_spacing, panorama.elevation_spacing])
    for i in range(panorama.Size[0]):
        for j in range(panorama.Size[1]):
            rho = panorama.PanoramaImage[i, j]
            if rho == panorama.void_data:
                filtered_image[i, j] = panorama.void_data
                continue
            elif np.isnan(rho):
                filtered_image[i, j] = np.nan
                continue
            else:
                # Create the filter and define the window size according to the changing resolution (sigma)
                current_sigma = 2.5 * phenomena_size / (rho * scan_resolution)
                ksize = np.ceil(2 * current_sigma + 1).astype(int)
                if ksize % 2 == 0:
                    ksize += 1

                ksizes.append(ksize)
                sigmas.append(current_sigma)

                gauss_kernel = cv2.getGaussianKernel(ksize, current_sigma)
                gauss_kernel = gauss_kernel.dot(gauss_kernel.T)

                # Define the window
                win_size = (ksize / 2).astype(int)
                i_start = max(0, i - win_size)
                i_start_win = max(0, win_size - i - i_start)
                i_end = min(panorama.Size[0], i + win_size + 1)
                i_end_win = min(gauss_kernel.shape[0], gauss_kernel.shape[0] - ((i + win_size) - i_end))

                j_start = max(0, j - win_size)
                j_start_win = max(0, win_size - j - j_start)
                j_end = min(panorama.Size[1], j + win_size + 1)
                j_end_win = min(gauss_kernel.shape[1], gauss_kernel.shape[1] - ((j + win_size) - j_end))

                patch = panorama.PanoramaImage[i_start:i_end , j_start: j_end ]
                gauss_win = gauss_kernel[i_start_win:i_end_win, j_start_win: j_end_win]

                non_nan = np.where(patch != panorama.void_data)
                filtered_image[i, j] = sum(patch[non_nan] * gauss_win[non_nan]) / sum(gauss_win[non_nan])

    sigma = np.mean(sigmas)
    ksize = np.mean(ksizes)
    return filtered_image, sigma, ksize


def computePanoramaDerivatives_adaptive(img, ksize=3, sigma=1., resolution=1., blur_window=(0,0), **kwargs):
    r"""
    Numerical and adaptive computation of central derivatives in x and y directions

    :param img: the panorama image
    :param ksize: window size in which the differentiation is carried
    :param sigma: sigma for gaussian blurring. Default: 1. If sigma=0 no smoothing is carried out
    :param blur_window: tuple of window size for blurring

    :type img: np.array
    :type ksize: int
    :type resolution: float
    :type sigma: float
    :type blur_window: tuple


    :return: tuple of the derivatives in the following order: :math:`(d_{\theta}, d_{\phi}, d__{\theta\theta}, d{\phi\phi}, d_{\theta\phi})`
    :rtype: tuple
    """

    # if blurring is required before differentiation
    import cv2
    img = (img).astype('float64')
    if sigma != 0:
        img = cv2.GaussianBlur(img, blur_window, sigma)

    # construct window sizes according to range
    # 1. window size at each cell
    d = np.ceil(ksize / (img * resolution)).astype('int')

    # 2. pad panorama according to cell size
    m, n = img.shape[:2]
    cols_indx = np.arange(0,n)
    rows_indx = np.arange(0,m)
    nn, mm = np.meshgrid(cols_indx, rows_indx) # indices of the panorama

    column_L = np.abs((nn - d).min()) # leftmost column
    column_R = (nn+ d).max()  # rightmost column
    row_T = np.abs((mm - d).min()) # topmost row
    row_B = (mm+d).max() # lowwermost row
    nn += column_L
    mm += row_T

    img_extended = boundPanoramaExpand(img, row_T, column_L, row_B, column_R)
    print('hello')

    # 3. compute differences for each window
    dx = np.zeros(img.shape)
    dy = np.zeros(img.shape)

    for window in tqdm(np.unique(d), desc='compute central derivatives'):
        indx = np.where(d==window)
        # central derivative
        dx[indx] = (img_extended[mm[indx] - window, nn[indx]] - img_extended[mm[indx] + window, nn[indx]]) / (2 * window * resolution)
        dy[indx] = (img_extended[mm[indx], nn[indx] - window] - img_extended[mm[indx], nn[indx]]+ window) / (2 * window * resolution)

    return dx, dy

def boundPanoramaExpand(A, num_rows_T=1, num_columns_L=1, num_rows_B=None, num_columns_R=None):
    """
    Expand the panorama using the respective columns from the other side of the panorama, for boundary condition

    Top and bottom boundaries are mirrored.

    :param A: the matrix to expand its boundaries
    :param num_rows_T: the number of rows to expand on the top
    :param num_columns_L: the number of columns to expand on the left
    :param num_rows_B: the number of rows to expand on the bottom. If not defined, will expand according by the same number as in the top (Default)
    :param num_columns_R: the number of columns to expand on the right. If not defined, will expand according by the same number as in the left (Default)

    :type A: np.array
    :type num_rows: int
    :type num_columns: int

    :return:  a re-adjusted matrix

    :rtype: np.array
    """
    m, n = A.shape[:2]

    if num_columns_R is None:
        num_columns_R = num_columns_L
    if num_rows_B is None:
        num_rows_B = num_rows_T


    B = np.zeros((m + num_rows_T + num_rows_B, n + num_columns_L + num_columns_R))
    B[num_rows_T:-num_rows_B, num_columns_L:-num_columns_R] = A

    # 1. left and right columns
    B[:, :num_columns_L] = B[:, -2 * num_columns_L : -num_columns_L] # copy right elements to the left
    B[ :, -num_columns_R:] = B[:, num_columns_R : 2* num_columns_R] # copy left elements to the right

    #2. top and bottom rows
    B[:num_rows_T, :] = np.flip(B[num_rows_T : 2* num_rows_T, : ], axis=0) # top
    B[-num_rows_B :, :] = np.flip(B[-2* num_rows_B:-num_rows_B, :], axis=0) #  bottom

    return B

def boundPanoramaShrink(A, num_rows_T=1, num_columns_L=1, num_rows_B=None, num_columns_R=None):
    """
     Shrink the panorama using to its original dimensions given the number of columns and rows previously added

    :param A: the matrix to expand its boundaries
    :param num_rows_T: the number of rows to expand on the top
    :param num_columns_L: the number of columns to expand on the left
    :param num_rows_B: the number of rows to expand on the bottom. If not defined, will expand according by the same number as in the top (Default)
    :param num_columns_R: the number of columns to expand on the right. If not defined, will expand according by the same number as in the left (Default)

    :type A: np.array
    :type num_rows: int
    :type num_columns: int

    :return:  a re-adjusted matrix

    :rtype: np.array
    """

    if num_columns_R is None:
        num_columns_R = num_columns_L
    if num_rows_B is None:
        num_rows_B = num_rows_T

    return A[num_rows_T:-num_rows_B, num_columns_L:-num_columns_R]


if __name__ == '__main__':
    A = np.array([[
        1, 2, 3, 11],
        [4, 5, 6, 12],
        [7, 8, 9, 13]])

    dx, dy = computePanoramaDerivatives_adaptive(A, sigma=0)

    import matplotlib.pyplot as plt
    plt.imshow(dx)
    plt.show()