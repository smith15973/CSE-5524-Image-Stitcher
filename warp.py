import numpy as np


def _apply_homography(H, pts):
    """
    Apply 3x3 homography H to array of 2D pts

    (x, y) = (col, row) order for pts

    H: (3, 3) 
    pts: (N, 2)

    Returns (N, 2) float array of transformed (x, y) pts
    """

    N = pts.shape[0]

    # Convert to homogeneous coords
    pts_h = np.hstack([pts, np.ones((N, 1))]) # (N, 3)

    # Matrix multiply, H at each column vector
    pts_out = (H @ pts_h.T).T # (N, 3)

    # Normalize by third coord -> (x, y)
    pts_out = pts_out[:, :2] / pts_out[:, 2:3]
    return pts_out


def compute_canvas_params(img_a, img_b, H):
    """
    Determine canvas size and offset needed for both images after
    warping img_b into img_a's coord. frame (Project img_b's four
    corners through H to see where they are in img_a's coord system.)

    Returns:
        canvas_h, canvas_w (canvas dimensions in pixels)
        offset_row, offset_col (where img_a's top-left corner sits on canvas)
    """
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    # Four corners of img_a in (x=col, y=row) order
    corners_a = np.array([
        [0,       0      ],
        [w_a - 1, 0      ],
        [0,       h_a - 1],
        [w_a - 1, h_a - 1],
    ], dtype=float)

    # Four corners of img_b in (x=col, y=row) order
    corners_b = np.array([
        [0,       0      ],
        [w_b - 1, 0      ],
        [0,       h_b - 1],
        [w_b - 1, h_b - 1],
    ], dtype=float)

    # Project img_b corners into img_a's coord frame using H
    corners_b_in_a = _apply_homography(H, corners_b)

    # Use img_a corners to find bounding box
    all_corners = np.vstack([corners_a, corners_b_in_a])  # (8, 2)

    min_x = all_corners[:, 0].min()   # left x
    min_y = all_corners[:, 1].min()   # top y
    max_x = all_corners[:, 0].max()   # right x
    max_y = all_corners[:, 1].max()   # bottom y

    # If bounding box goes negative, shift everything right/down
    # offset_col/offset_row = how many pixels img_a is pushed from (0,0)
    offset_col = int(np.floor(-min_x)) if min_x < 0 else 0
    offset_row = int(np.floor(-min_y)) if min_y < 0 else 0

    # Canvas size = full bounding box +1 for max pixel
    canvas_w = int(np.ceil(max_x)) + offset_col + 1
    canvas_h = int(np.ceil(max_y)) + offset_row + 1

    return canvas_h, canvas_w, offset_row, offset_col


def _bilinear_sample(img, y_coords, x_coords):
    """
    Sample image at fractional coords using bilinear interpolation
    for smoother result than nearest-neighbor sampling

    For frational coord (x, y), find four surrounding int pixel 
    corners and compute weighted average

    (1-fy)(1-fx)*top_left + (1-fy)*fx*top_right + fy*(1-fx)*bot_left + fy*fx*bot_right

    (fx = x - floor(x), fy = y - floor(y))

    img: (H, W, C) (src img)
    y_coords: (H_out, W_out)  (fractional row indices into img)
    x_coords: (H_out, W_out) (fractional col indices into img)

    Returns (H_out, W_out, C) sampled array.
    """
    h, w = img.shape[:2]

    # Int corners around each fractional coordinate
    y0 = np.floor(y_coords).astype(int)
    x0 = np.floor(x_coords).astype(int)
    y1 = y0 + 1
    x1 = x0 + 1

    # Fractional distance to +1 neighbor (0 = at corner, 1 = at next corner)
    fy = (y_coords - y0)[:, :, np.newaxis]   # (H_out, W_out, 1)
    fx = (x_coords - x0)[:, :, np.newaxis]

    # Clamp indices to valid img bounds
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)

    # Sample four surrounding pixels
    top_left  = img[y0, x0]   # (H_out, W_out, C)
    top_right = img[y0, x1]
    bot_left  = img[y1, x0]
    bot_right = img[y1, x1]

    # Weighted combination, bilinear interpolation
    out = ((1 - fy) * (1 - fx) * top_left
         + (1 - fy) *      fx  * top_right
         +      fy  * (1 - fx) * bot_left
         +      fy  *      fx  * bot_right)

    return out


def warp_images_onto_canvas(img_a, img_b, H):
    """
    Put both images onto shared canvas

    img_a placed directly, img_b inverse-warped into img_a's frame using H.
    (Forward warping (map each img_b pixel → canvas) leaves gaps b/c
    homography maps continuous to discrete grid unevenly)

    H: (3,3) float array (maps img_b coords to img_a coords: p_a ≈ H @ p_b)

    Note: Inputs must be RGB: (H, W, 3) uint8 or float.

    Returns:
        canvas_a:         (canvas_h, canvas_w, 3) float (img_a on canvas)
        canvas_b:         (canvas_h, canvas_w, 3) float (warped img_b on canvas)
        mask_a:           (canvas_h, canvas_w)    float (1.0 where canvas_a is valid)
        mask_b:           (canvas_h, canvas_w)    float (1.0 where canvas_b is valid)
        offset:           (offset_row, offset_col) — where img_a's origin sits
    """
    canvas_h, canvas_w, offset_row, offset_col = compute_canvas_params(img_a, img_b, H)

    # Zero-filled canvases + validity masks
    canvas_a = np.zeros((canvas_h, canvas_w, 3), dtype=float)
    canvas_b = np.zeros((canvas_h, canvas_w, 3), dtype=float)
    mask_a   = np.zeros((canvas_h, canvas_w),    dtype=float)
    mask_b   = np.zeros((canvas_h, canvas_w),    dtype=float)

    # Put img_a on canvas
    h_a, w_a = img_a.shape[:2]
    r0_a, r1_a = offset_row, offset_row + h_a
    c0_a, c1_a = offset_col, offset_col + w_a

    canvas_a[r0_a:r1_a, c0_a:c1_a] = img_a.astype(float)
    mask_a[r0_a:r1_a, c0_a:c1_a]   = 1.0

    # Inverse-warp img_b onto the canvas
    #
    # H maps img_b → img_a, take reverse:  H_inv maps img_a → img_b.
    # Given canvas pixel, subtract offset to get img_a coords, then apply H_inv to get img_b coords.
    H_inv = np.linalg.inv(H)

    # Build grid of (col, row) = (x, y) coords for every canvas pixel
    cols = np.arange(canvas_w)
    rows = np.arange(canvas_h)
    cc, rr = np.meshgrid(cols, rows) # cc[r,c]=c (x), rr[r,c]=r (y)

    # Convert canvas coords to img_a coords by subtracting offset
    x_a = (cc - offset_col).astype(float) # horizontal (col) in img_a frame
    y_a = (rr - offset_row).astype(float) # vertical (row) in img_a frame

    # Stack into homogeneous vectors, shape (3, canvas_h * canvas_w)
    ones  = np.ones_like(x_a)
    pts_h = np.stack([x_a.ravel(), y_a.ravel(), ones.ravel()], axis=0)  # (3, N)

    # Apply H_inv to map every img_a coord to img_b coord
    pts_b_h = H_inv @ pts_h        # (3, N)  — homogeneous img_b coords

    # Normalize homogeneous coord to get Cartesian (x, y)
    pts_b_h /= pts_b_h[2:3, :]

    x_b = pts_b_h[0].reshape(canvas_h, canvas_w) # fractional col in img_b
    y_b = pts_b_h[1].reshape(canvas_h, canvas_w) # fractional row in img_b

    # Find which canvas pixels are inside img_b (outside = 0)
    h_b, w_b = img_b.shape[:2]
    valid = (
        (x_b >= 0) & (x_b <= w_b - 1) &
        (y_b >= 0) & (y_b <= h_b - 1)
    )

    # Bilinear-interpolate img_b at fractional coords
    sampled = _bilinear_sample(img_b.astype(float), y_b, x_b) # (canvas_h, canvas_w, 3)

    # Write only valid pixels to canvas
    canvas_b[valid] = sampled[valid]
    mask_b[valid]   = 1.0

    return canvas_a, canvas_b, mask_a, mask_b, (offset_row, offset_col)
