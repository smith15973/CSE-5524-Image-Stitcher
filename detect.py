import cv2
import numpy as np

def detect_keypoints(image, debug: bool = False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create()
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    if debug:
        img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, (0, 0, 255))
        cv2.imshow("Keypoints", img_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Convert KeyPoint objects to an (N, 2) array of (x, y) coordinates
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return points, descriptors
