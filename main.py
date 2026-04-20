
import sys
import cv2
import numpy as np
from detect import detect_keypoints
from match import match_keypoints, show_matches


def load_image(path: str, max_dim: int = 1600, debug: bool = False):
    image = cv2.imread(path)
    if image is None:
        sys.exit(f"Could not read {path}")
    

    # downsample the image to allow for quicker matching nad avoid memory issues
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    
    if debug:
        cv2.imshow("Loaded Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image

def stitch_images(left_image, right_image, debug: bool = False):
    keypoints_left, descriptors_left = detect_keypoints(left_image, debug=debug)
    keypoints_right, descriptors_right = detect_keypoints(right_image, debug=debug)

    print(f"Left Interesting Points: {len(keypoints_left)}")
    print(f"Right Interesting Points: {len(keypoints_right)}")

    matches_l, matches_r = match_keypoints(
        keypoints_left=keypoints_left,
        keypoints_right=keypoints_right,
        descriptors_left=descriptors_left,
        descriptors_right=descriptors_right
        )
    
    if debug:
        show_matches(left_image, right_image, matches_l, matches_r)

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <left.jpg> <right.jpg> <output.jpg>")
        sys.exit(1)
    debug = True  # Set to True to visualize keypoints
    left_image = load_image(sys.argv[1], debug=debug)
    right_image = load_image(sys.argv[2], debug=debug)
    panorama = stitch_images(left_image, right_image, debug=debug)

if __name__ == "__main__":
    main()