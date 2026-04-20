import numpy as np

# Gaussian kernel from Burt & Adelson
# 1D form: w = [0.25 - a/2,  0.25,  a,  0.25,  0.25 - a/2]
# For a = 0.375: w = [0.0625, 0.25, 0.375, 0.25, 0.0625]
_GAUSS_1D = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])


# Convolution helpers
def _convolve_rows(img, kernel):
    """
    Convolve 2D image w/ 1D kernel applied horizontally, use reflected padding at borders to avoid edge issues
    """

    k = len(kernel) // 2  # half-width of kernel

    # Pad left and right by k pixels, mirroring imag edge
    padded = np.pad(img, ((0, 0), (k, k)), mode='reflect')
    out = np.zeros_like(img, dtype=float)

    # Slide kernel across each position and add to weighted sum
    for i, w in enumerate(kernel):
        out += w * padded[:, i : i + img.shape[1]]

    return out


def _convolve_cols(img, kernel):
    """
    Convolve 2D image with a 1D kernel applied vertically (along rows) w/ reflected padding at borders
    """
    k = len(kernel) // 2

    padded = np.pad(img, ((k, k), (0, 0)), mode='reflect')
    out = np.zeros_like(img, dtype=float)

    for i, w in enumerate(kernel):
        out += w * padded[i : i + img.shape[0], :]

    return out


def _gaussian_blur(img, kernel=_GAUSS_1D):
    """
    Blur single-channel 2D image using separable Gaussian kernel.
    """
    blurred = _convolve_rows(img, kernel)
    blurred = _convolve_cols(blurred, kernel)
    return blurred


# For Img Pyramids:

def _reduce(img):
    """
    Downsample img by factor of 2. Blur first, then subsample (keep only even-indexed rows and cols)
    """
    blurred = _gaussian_blur(img)
    return blurred[::2, ::2]


def _expand(img, target_shape=None):
    """
    Upsample img by factor of 2. Insert zeros between every pixel, Convolve with 2 * Gaussian kernel to fill gaps

    target_shape: if given, result cropped to (H, W) (reduce/expand on odd-dimension imag can be off by 1)
    """ 

    h, w = img.shape

    # Create zero-filled grid, put original pixels at even positions
    expanded = np.zeros((h * 2, w * 2), dtype=float)
    expanded[::2, ::2] = img

    # Convolve, fill in zero gaps with interpolation
    expanded = _gaussian_blur(expanded, kernel=2.0 * _GAUSS_1D)

    # Trim to match target shape if input had odd dimensions
    if target_shape is not None:
        expanded = expanded[: target_shape[0], : target_shape[1]]

    return expanded

def build_gaussian_pyramid(img, levels):
    """
    Builds a Gaussian pyramid, Level 0 is original img.

    img: 2D float array (single channel)
    levels: total # of levels including original

    Returns a list of 'levels' 2D arrays
    """

    pyramid = [img.astype(float)]

    for _ in range(levels - 1):
        pyramid.append(_reduce(pyramid[-1]))

    return pyramid


def build_laplacian_pyramid(img, levels):
    """
    Builds a Laplacian (detail/error) pyramid.

    Each level stores detail lost between levels: L[k] = G[k]  -  expand(G[k+1])

    Note: top (coarsest) level stored as-is, nothing to subtract

    img: 2D float array (single channel)
    levels: # of pyramid levels

    Returns list of 'levels' 2D arrays
    """

    gauss_pyr = build_gaussian_pyramid(img, levels)
    lap_pyr = []

    for k in range(levels - 1):
        # Upsample next coarser level back to this level's size
        expanded = _expand(gauss_pyr[k + 1], target_shape=gauss_pyr[k].shape)

        # Detail lost
        lap_pyr.append(gauss_pyr[k] - expanded)

    # Use coarsest level as top of pyramid
    lap_pyr.append(gauss_pyr[-1])

    return lap_pyr


def collapse_laplacian_pyramid(lap_pyr):
    """
    Reconstructs imag from Laplacian pyramid. Upsample and add detial back in: G[k] = expand(G[k+1]) + L[k]

    lap_pyr: list of 2D arrays (output of build_laplacian_pyramid)

    Returns reconstructed 2D float array
    """

    # Start from the top level
    img = lap_pyr[-1].copy()

    # Go down pyramid, adding detail at each finer level
    for k in range(len(lap_pyr) - 2, -1, -1):
        # Upsample to match next level size
        img = _expand(img, target_shape=lap_pyr[k].shape)

        # Add back detail stored at this level
        img = img + lap_pyr[k]

    return img
