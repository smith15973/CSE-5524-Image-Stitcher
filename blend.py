import numpy as np
from pyramids import build_laplacian_pyramid, build_gaussian_pyramid, collapse_laplacian_pyramid


# Blend weight/seam mask helpers

def _make_soft_blend_weight(mask_a, mask_b):
    """
    Builds a per-pixel float weight for canvas_a (values [0, 1]), pixels covered only by img_a → weight 1.0  (fully A),
    Pixels covered only by img_b  → weight 0.0  (fully B), Overlap pixels-> 1.0 at the left edge of overlap, 0.0 at the right edge

    Note: uses left -> right panorama layout. Weight for canvas_b = 1.0 - weight_a

    mask_a, mask_b: (H, W) binary float arrays from warp_images_onto_canvas
    Returns (H, W) float array.
    """
    h, w = mask_a.shape
    overlap = (mask_a > 0) & (mask_b > 0)

    # Put 1.0 everywhere img_a is valid, 0.0 else
    weight_a = mask_a.copy().astype(float)

    if not overlap.any():
        # No overlapping region = nothing to ramp
        return weight_a

    # Find which cols have overlap pixels
    col_has_overlap = overlap.any(axis=0)   # (W,) bool
    overlap_cols    = np.where(col_has_overlap)[0]
    c_left  = overlap_cols[0]
    c_right = overlap_cols[-1]

    # Build 1D ramp, 1.0 at the left boundary, 0.0 at the right boundary
    n_cols = c_right - c_left + 1
    ramp_1d = np.linspace(1.0, 0.0, n_cols)   # (n_cols,)

    # Ramp -> full (H, W) array
    ramp_2d = np.ones((h, w), dtype=float)
    ramp_2d[:, c_left : c_right + 1] = ramp_1d  # same ramp for every row
    ramp_2d[:, c_right + 1 :] = 0.0  # past overlap: fully B

    # Apply ramp only inside the overlap region
    weight_a[overlap] = ramp_2d[overlap]

    return weight_a


def _make_hard_seam_mask(mask_a, mask_b):
    """
    Build binary blend mask that splits the overlap at its center col. Left half of overlap (and exclusive-A region) → 1.0
    (A territory), Right half of overlap (and exclusive-B region) → 0.0  (B territory)

    Returns (H, W) float array (values are 0.0 or 1.0)
    """
    h, w = mask_a.shape
    overlap = (mask_a > 0) & (mask_b > 0)

    # Put 1.0 everywhere img_a has a pixel
    blend_mask = mask_a.copy().astype(float)

    if not overlap.any():
        return blend_mask

    # Seam = center column of the overlap region
    col_has_overlap = overlap.any(axis=0)
    overlap_cols    = np.where(col_has_overlap)[0]
    seam_col = (overlap_cols[0] + overlap_cols[-1]) // 2

    # Build boolean array, True for every col right of seam
    col_indices   = np.arange(w) # (W,)
    right_of_seam = col_indices[np.newaxis, :] > seam_col # (1, W) -> (H, W)

    # In the overlap, pixels right of the seam belong to img_b (weight_a = 0)
    blend_mask[overlap & right_of_seam] = 0.0

    return blend_mask


# Blending methods

def _alpha_blend(canvas_a, canvas_b, mask_a, mask_b):
    """
    Simple weighted average blend. Overlap region feathered with linear alpha ramp: img_a fades
    out and img_b fades in moving left-to-right across  overlap.

    canvas_a, canvas_b: (H, W, 3)
    mask_a, mask_b:     (H, W) (1.0 = valid, 0.0 = empty)

    Returns (H, W, 3) float blended image
    """

    # Find per-pixel weight for canvas_a (0 to 1); B weight = 1 - weight_a
    weight_a = _make_soft_blend_weight(mask_a, mask_b)  # (H, W)
    weight_b = 1.0 - weight_a

    # Expand to (H, W, 1) for 3 color channels
    w_a = weight_a[:, :, np.newaxis]
    w_b = weight_b[:, :, np.newaxis]

    result = w_a * canvas_a + w_b * canvas_b
    return result


def _pyramid_blend(canvas_a, canvas_b, mask_a, mask_b, levels=4):
    """
    Multi-resolution Laplacian pyramid blend (Burt & Adelson). Blend low-frequency content
    (big structures) with wide transition zone and high-frequency content (fine details) with
    narrow transition zone (hides seams across all scales)

    - Split each image into a Laplacian pyramid
    - Build Gaussian pyramid of a binary seam mask
    - At every pyramid level, blend: blended[k] = mask[k]*A[k] + (1-mask[k])*B[k]
    - Collapse blended Laplacian pyramid to reconstruct final image.

    Note: The color channels are processed independently (same mask for all)

    canvas_a, canvas_b: (H, W, 3)
    mask_a, mask_b:     (H, W) 
    levels:             number of pyramid levels (more levels = smoother blend but slower and requires larger images)

    Returns (H, W, 3) float blended image.
    """

    # Make hard binary seam mask
    seam_mask = _make_hard_seam_mask(mask_a, mask_b) # (H, W), values 0 or 1

    # Build Gaussian pyramid of the mask. Each coarser level = blurred
    # further -> wider + smoother gradient (blend weight at that scale)
    mask_pyr = build_gaussian_pyramid(seam_mask, levels) # list of (H_k, W_k)

    # Process each color channel separately
    blended_channels = []
    for c in range(3):
        channel_a = canvas_a[:, :, c] # (H, W) single channel
        channel_b = canvas_b[:, :, c]

        # Build Laplacian pyramids
        lap_a = build_laplacian_pyramid(channel_a, levels)
        lap_b = build_laplacian_pyramid(channel_b, levels)

        # Blend at every pyramid level using mask at that level
        blended_lap = []
        for k in range(levels):
            m = mask_pyr[k] # blend weight at scale k: (H_k, W_k)

            # Linear blend: A where mask=1, B where mask=0, gradient in between
            blended_lap.append(m * lap_a[k] + (1.0 - m) * lap_b[k])

        # Collapse blended Laplacian pyramid back to a full image
        blended_channels.append(collapse_laplacian_pyramid(blended_lap))

    # Recombine three blended channels into single RGB image
    result = np.stack(blended_channels, axis=2) # (H, W, 3)
    return result

def blend(canvas_a, canvas_b, mask_a, mask_b, method='alpha', levels=4):
    """
    Blend two images-on-canvas into single panoramic result.

    Params
    canvas_a, canvas_b : (H, W, 3)
    mask_a, mask_b : (H, W) (Validity masks, 1.0 where canvas has actual img data, 0.0 where empty)

    Method : Blur method (alpha = simple, pyramid = better)

    levels : # of pyramid levels if using 'pyramid' blend

    Returns
    -------
    (H, W, 3) float array (blended panoramic image)
    """

    if method == 'alpha':
        return _alpha_blend(canvas_a, canvas_b, mask_a, mask_b)
    elif method == 'pyramid':
        return _pyramid_blend(canvas_a, canvas_b, mask_a, mask_b, levels=levels)
    else:
        raise ValueError(f"Unknown blend method '{method}'. Choose 'alpha' or 'pyramid'.")
