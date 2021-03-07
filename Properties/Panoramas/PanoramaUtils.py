import numpy as np
from tqdm import tqdm


def computePanoramaDerivatives_adaptive(img, ksize=0.2, sigma=0., resolution=1., blur_window=(0,0), rangeImage=None):
    r"""
    Numerical and adaptive computation of central derivatives in x and y directions

    :param img: the panorama image
    :param ksize: world window size (meters). Default: 0.2 m
    :param sigma: sigma for gaussian blurring. Default: 0. If sigma=0 no smoothing is carried out
    :param resolution: scanning resolution (mean value between elevation and azimuth resolution)
    :param blur_window: tuple of window size for blurring
    :param rangeImage: if sent, the window size will be computed according to the rangeImage and not according to the img


    :type img: np.array
    :type ksize: float
    :type resolution: float
    :type sigma: float
    :type blur_window: tuple


    :return: tuple of the derivatives in the following order: (d_{\theta}, d_{\phi}, d__{\theta\theta}, d{\phi\phi}, d_{\theta\phi})
    :rtype: tuple
    """

    # if blurring is required before differentiation
    import cv2
    import Utils.MyTools as mt
    img = (img).astype('float64')
    if sigma != 0:
        img = cv2.GaussianBlur(img, blur_window, sigma)

    if rangeImage is None:
        rangeImage = img

    # construct window sizes according to range
    # 1. window size at each cell
    d = np.ceil(ksize / (rangeImage * resolution)).astype('int')

    # 2. pad panorama according to cell size
    m, n = img.shape[:2]
    cols_indx = np.arange(0,n)
    rows_indx = np.arange(0,m)
    nn, mm = np.meshgrid(cols_indx, rows_indx) # indices of the panorama

    column_L = np.abs((nn - d).min()) # leftmost column
    column_R = (nn+ d).max()  - n + 1# rightmost column
    row_T = np.abs((mm - d).min()) # topmost row
    row_B = (mm+d).max() - m + 1# lowwermost row
    nn += column_L
    mm += row_T

    img_extended = boundPanoramaExpand(img, row_T, column_L, row_B, column_R)
    #
    # 3. compute differences for each window
    dx = np.zeros(img.shape)
    dy = np.zeros(img.shape)

    for window in tqdm(np.unique(d), desc='compute central derivatives'):
        indx = np.where(d==window)
        # central derivative
        dx_, dy_ = mt.imageDerivatives_4connected(img_extended, 1, window, sigma, resolution, blur_window=blur_window)
        dx_ = dx_[row_T:-row_B, column_L:-column_R]
        dy_ = dy_[row_T:-row_B, column_L:-column_R]
        dx[indx] = dx_[indx]
        dy[indx] = dy_[indx]

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

    # 2. top and bottom rows
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