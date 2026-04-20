import numpy as np
import matplotlib.pyplot as plt

from warp import warp_images_onto_canvas
from blend import blend


def stitch(img_a, img_b, H, blend_method='alpha', pyramid_levels=4, debug=False):
    """
    Stitch two overlapping images into a panorama

    img_a is reference image, img_b warped into img_a's coord frame using homography H, then blend

    Params:
    img_a : (H_a, W_a, 3) (reference (left) image)
    img_b : (H_b, W_b, 3)  (imag to warp (right))
    H     : (3, 3) float (homography mapping img_b coords → img_a coords)

    ***Blend_method : change 'alpha' for basic blend, 'pyramid' for multi-scale blend***

    pyramid_levels : int  (— )pyramid depth for 'pyramid' method)

    debug : if True, display intermediate results before blending:
        - canvas_a  (img_a put on shared canvas)
        - canvas_b  (img_b warped onto canvas)
        - mask_a    (validity mask for canvas_a)
        - mask_b    (validity mask for canvas_b)

    Returns (canvas_h, canvas_w, 3) array (stitched panorama)
    """

    # Warp both images onto a shared canvas
    # canvas_a  (img_a copied to correct position, zeros elsewhere)
    # canvas_b  (img_b inverse-warped to align with img_a, zeros elsewhere)
    # mask_a/b  (1.0 where each canvas has real img data)
    canvas_a, canvas_b, mask_a, mask_b, offset = warp_images_onto_canvas(img_a, img_b, H)

    if debug:
        _show_debug_canvases(canvas_a, canvas_b, mask_a, mask_b)

    # Blend the two aligned canvases
    # CHANGE BLEND METHOD HERE!!***
    blended = blend(canvas_a, canvas_b, mask_a, mask_b,
                    method=blend_method, levels=pyramid_levels)

    # Clip to [0, 255], convert to uint8 for display
    result = np.clip(blended, 0, 255).astype(np.uint8)

    return result


# Debug helper function
def _show_debug_canvases(canvas_a, canvas_b, mask_a, mask_b):
    """
    Display both canvases and their masks side-by-side
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Pass img_a[:,:,::-1] to stitch() for correct matplotlib colors.
    axes[0, 0].imshow(canvas_a.astype(np.uint8))
    axes[0, 0].set_title("canvas_a  (img_a, no transform)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(canvas_b.astype(np.uint8))
    axes[0, 1].set_title("canvas_b  (img_b warped into img_a frame)")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(mask_a, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title("mask_a  (white = valid pixels)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_b, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title("mask_b  (white = valid pixels)")
    axes[1, 1].axis('off')

    plt.suptitle("Debug: canvas contents before blending", fontsize=13)
    plt.tight_layout()
    plt.show()
