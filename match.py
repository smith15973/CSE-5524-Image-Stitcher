import numpy as np
import cv2

# based on slide 73 of InterestPoints notes
def match_keypoints(keypoints_left, descriptors_left, keypoints_right, descriptors_right, debug: bool = False):
    matches = []

    counter = 0
    for i in range(len(descriptors_left)):
        # Original implementation: inner loop over every descriptor in the right image
        # distances = []
        # for j in range(len(descriptors_right)):
        #     dist = np.linalg.norm(descriptors_left[i] - descriptors_right[j])
        #     distances.append(dist)
        #
        # This is correct but slow: for every descriptor in the left image,

        # Optimization: vectorize the inner loop with broadcasting 
        distances = np.linalg.norm(descriptors_right - descriptors_left[i], axis=1)

        sorted_distances = np.argsort(distances)
        best_j = sorted_distances[0]
        second_j = sorted_distances[1]

        # Lowe's ratio test — 0.8 is the threshold recommended in the original SIFT paper
        if distances[best_j] < 0.8 * distances[second_j]:
            matches.append((i, best_j))

        counter += 1

        print(f"\r{len(matches)}/{counter}", end="", flush=True)

    print()

    matches = np.array(matches)
    matched_l = keypoints_left[matches[:, 0]]
    matched_r = keypoints_right[matches[:, 1]]
    print(f"Matches: {len(matches)}")
    return matched_l, matched_r



# ***Visualization Help from Claude
def show_matches(image_l, image_r, matches_l, matches_r, max_lines=50):
    # Build a side-by-side canvas sized to the taller image
    h_l, w_l = image_l.shape[:2]
    h_r, w_r = image_r.shape[:2]
    canvas_h = max(h_l, h_r)
    canvas_w = w_l + w_r
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Paste both images in at y=0
    canvas[:h_l, :w_l] = image_l
    canvas[:h_r, w_l:w_l + w_r] = image_r

    # Optionally downsample to keep the visualization readable
    n = len(matches_l)
    if max_lines is not None and n > max_lines:
        idx = np.random.choice(n, size=max_lines, replace=False)
    else:
        idx = np.arange(n)

    # Draw each match: circle on each side + line between them
    for k in idx:
        x_l, y_l = matches_l[k]
        x_r, y_r = matches_r[k]
        pt_l = (int(x_l), int(y_l))
        pt_r = (int(x_r) + w_l, int(y_r))   # shift right-image x by left-image width

        cv2.circle(canvas, pt_l, 4, (0, 255, 0), 1)
        cv2.circle(canvas, pt_r, 4, (0, 255, 0), 1)
        cv2.line(canvas, pt_l, pt_r, (0, 255, 255), 1)

    cv2.imshow("Matches", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    