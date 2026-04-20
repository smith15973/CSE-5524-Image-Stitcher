import numpy as np
import cv2

# Used homography implementation from Homework 9

def normalize_points(pts):
    # Condition 1: Compute centroid so we can give the points zero mean
    centroid = pts.mean(axis=0)          # (2,)

    # Shift points so they have zero mean
    shifted = pts - centroid

    # Condition 2: Compute average distance of shifted points to origin
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))

    # Scale factor s so that average distance of shifted points to origin = sqrt(2)
    s = np.sqrt(2) / mean_dist

    # Build similarity transformation matrix T
    T = np.array([
        [s,  0, -s * centroid[0]],
        [0,  s, -s * centroid[1]],
        [0,  0,  1              ]
    ])

    # Apply T to points by converting to homogeneous coordinates first
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])   # (N, 3)
    pts_norm = (T @ pts_h.T).T                          # (N, 3)

    # Return inhomogeneous normalized points and T
    return pts_norm[:, :2], T

def calc_homography(pts1, pts2):
    # --- Normalize both point sets ---
    N = len(pts1)
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)


    # Build matrix A (2N x 9)
    A = []
    for i in range(N):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]

        # Row 1:  [x1, y1, 1, | 0, 0, 0, | -x1*x1', -y1*x1', -x1']
        row1 = [x1, y1, 1,
                0, 0, 0,
                -x1*x2, -y1*x2, -x2]

        # Row 2:  [0, 0, 0, | x1, y1, 1,| -x1*y1', -y1*y1', -y1']
        row2 = [0, 0, 0,
                x1, y1, 1,
                -x1*y2, -y1*y2, -y2]

        A.append(row1)
        A.append(row2)

    A = np.array(A)   # (2N, 9)

    # Ah = 0
    # A^TAh = 0
    ATA = A.T @ A

    # solve for eigenvalues and eigenvectors
    e_values, e_vectors = np.linalg.eig(ATA)

    # find index of smallest eigenvalue
    min_idx = np.argmin(e_values)

    # Get eigenvector corresponding to the smallest eigenvalue
    h = e_vectors[:, min_idx]

    # Reshape into 3x3 homography matrix
    H = h.reshape(3, 3)

    # Denormalize
    H = np.linalg.inv(T2) @ H @ T1

    H = H / H[2, 2]

    return H

def project_points(H, pts):
    """Apply homography H to an (N, 2) array of points. Returns (N, 2)."""
    N = len(pts)
    pts_h = np.hstack([pts, np.ones((N, 1))])     # (N, 3)
    proj_h = (H @ pts_h.T).T                       # (N, 3)
    # Guard against division by zero (shouldn't happen for sane Hs, but RANSAC
    # occasionally samples degenerate 4-tuples)
    w = proj_h[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return proj_h[:, :2] / w


# From Image Registration notes slides 19
def ransac_homography(src, dst, n=4, d=4.0, T_frac=0.7, N=2000, seed=0):
    rng = np.random.default_rng(seed)
    M = len(src)
    T = int(T_frac * M)

    best_inliers = np.zeros(M, dtype=bool)
    best_count = 0

    for i in range(N):
        # Step 1: sample n points (4 for homography) and estimate H
        idx = rng.choice(M, n, replace=False)
        try:
            H = calc_homography(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(H).all():
            continue

        # Step 2: determine consensus set (points within distance d)
        projected = project_points(H, src)
        if not np.isfinite(projected).all():
            continue
        errors = np.linalg.norm(projected - dst, axis=1)
        inliers = errors < d
        count = int(inliers.sum())

        # Track best
        if count > best_count:
            best_count = count
            best_inliers = inliers

        # Step 3: early termination if we have "enough" inliers
        if count >= T:
            break

    if best_count < 4:
        raise RuntimeError("RANSAC failed — no model had 4+ inliers")

    # Step 3/5: re-estimate H using all inliers from best consensus set
    H_refined = calc_homography(src[best_inliers], dst[best_inliers])
    return H_refined, best_inliers


def show_homography(left_image, H, matches_l, matches_r, max_points=50):
    """
    Visualize a homography by projecting right-image matched keypoints through H
    and overlaying them on the left image alongside the actual left-image keypoints.

    - Green dots: actual matched points in the left image
    - Red dots:   right-image points projected through H (should land on the green dots)
    - Yellow lines: residual error between each green/red pair
    """
    canvas = left_image.copy()

    # Project right-image points through H into left's coordinate frame    
    projected = project_points(H=H, pts=matches_r)

    # Downsample if there are too many points to look at clearly
    N = len(matches_r)
    if max_points is not None and N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
    else:
        idx = np.arange(N)

    for k in idx:
        actual = (int(matches_l[k, 0]), int(matches_l[k, 1]))
        proj = (int(projected[k, 0]), int(projected[k, 1]))

        cv2.line(canvas, actual, proj, (0, 255, 255), 1)      # yellow residual line
        cv2.circle(canvas, actual, 4, (0, 255, 0), -1)         # green actual
        cv2.circle(canvas, proj, 4, (0, 0, 255), -1)           # red projected

    # Report mean reprojection error in pixels — single most useful number
    errors = np.linalg.norm(projected - matches_l, axis=1)
    print(f"Mean reprojection error: {errors.mean():.2f} px")
    print(f"Median reprojection error: {np.median(errors):.2f} px")
    print(f"Max reprojection error: {errors.max():.2f} px")

    cv2.imshow("Homography check", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return canvas