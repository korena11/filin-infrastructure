import numpy as np



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


