
import sys
import cv2
from detect import sift_detect_keypoints, harris_detect_keypoints, describe_points, show_keypoints
from match import match_keypoints, show_matches
from homography import ransac_homography, show_homography
from stitch import stitch as warp_and_blend


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
    keypoints_left, descriptors_left = sift_detect_keypoints(left_image)
    keypoints_right, descriptors_right = sift_detect_keypoints(right_image)
    
    # keypoints_left = harris_detect_keypoints(left_image)
    # keypoints_right = harris_detect_keypoints(right_image)
    # keypoints_left, descriptors_left = describe_points(left_image, keypoints_left)
    # keypoints_right, descriptors_right = describe_points(right_image, keypoints_right)

    show_keypoints(left_image, keypoints_left, "Left Image Keypoints")
    show_keypoints(right_image, keypoints_right, "Right Image Keypoints")



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

    H, inliers = ransac_homography(matches_r, matches_l)   # note: r, l order
    if debug:
        show_homography(
            left_image=left_image,
            H=H,
            matches_l=matches_l,
            matches_r=matches_r
        )

    # Note for color channels: for matplotlib (plt.imshow), convert first: plt.imshow(panorama[:, :, ::-1])
    # ***CHANGE BLEND METHOD HERE!!***
    # 'alpha' = simple, 'pyramid' = multi-scale, smoother
    panorama = warp_and_blend(left_image, right_image, H,
                              blend_method='pyramid', debug=debug)
    return panorama


def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <left.jpg> <right.jpg> <output.jpg>")
        sys.exit(1)
    left_image  = load_image(sys.argv[1])
    right_image = load_image(sys.argv[2])
    output_path = sys.argv[3]

    panorama = stitch_images(left_image, right_image, debug=True)

    # Save and display result
    if panorama is not None:
        cv2.imwrite(output_path, panorama)
        print(f"Saved panorama to {output_path}")
        cv2.imshow("Panorama", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()