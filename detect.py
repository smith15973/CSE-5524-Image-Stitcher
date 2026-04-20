import cv2
import numpy as np

def sift_detect_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create()
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    # Convert KeyPoint objects to an (N, 2) array of (x, y) coordinates
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return points, descriptors


def harris_detect_keypoints(image,
                  sigma_d: float = 1.0,
                  sigma_i: float = 1.5,
                  alpha: float = 0.04,
                  threshold_ratio: float = 0.01,
                  nms_radius: int = 5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Slide 24
    smoothed = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=sigma_d)
    Ix = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Slide 25: Gaussian window
    Sxx = cv2.GaussianBlur(Ixx, ksize=(0, 0), sigmaX=sigma_i)
    Syy = cv2.GaussianBlur(Iyy, ksize=(0, 0), sigmaX=sigma_i)
    Sxy = cv2.GaussianBlur(Ixy, ksize=(0, 0), sigmaX=sigma_i)

    # Slide 29: Harris corner response 
    # R = det(M) - alpha * trace(M)^2
    # For a 2x2 M = [[Sxx, Sxy], [Sxy, Syy]]:
    #   det(M)   = Sxx * Syy - Sxy * Sxy
    #   trace(M) = Sxx + Syy
    det_M = Sxx * Syy - Sxy * Sxy
    trace_M = Sxx + Syy
    R = det_M - alpha * (trace_M ** 2)

    # Slide 30: threshold small R values, then non-max suppression
    threshold = threshold_ratio * R.max()
    R_thresh = np.where(R > threshold, R, 0)

    # Non-max suppression via a max-pool: a pixel is a local max iff its R
    # equals the max R in its (2*nms_radius+1) neighborhood.
    kernel_size = 2 * nms_radius + 1
    local_max = cv2.dilate(R_thresh,
                           np.ones((kernel_size, kernel_size), dtype=np.uint8))
    is_peak = (R_thresh == local_max) & (R_thresh > 0)

    # Extract (x, y) from (row, col) of peaks
    ys, xs = np.nonzero(is_peak)
    points = np.stack([xs, ys], axis=1).astype(np.float32)

    return points


# Describe each keypoint with a normalized patch (slide 68)
def describe_points(image, points, patch_size: int = 15):
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    half = patch_size // 2
    h, w = gray_image.shape
    # Patches too close to the edge get ignored
    valid = (points[:, 0] >= half) & (points[:, 0] < w - half) & \
            (points[:, 1] >= half) & (points[:, 1] < h - half)
    points = points[valid]

    descriptors = np.zeros((len(points), patch_size * patch_size), dtype=np.float32)
    for i, (x, y) in enumerate(points):
        xi, yi = int(x), int(y)
        patch = gray_image[yi - half: yi + half + 1,
                     xi - half: xi + half + 1].astype(np.float32)

        # Normalize
        patch = patch - patch.mean()
        std = patch.std()
        if std > 1e-6:
            patch = patch / std

        descriptors[i] = patch.flatten()

    return points, descriptors



def show_keypoints(image, points, label = "Keypoints"):
    vis = image.copy()
    for (x, y) in points:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
    cv2.imshow(label, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()