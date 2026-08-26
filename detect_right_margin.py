
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from detect_left_margin import get_reader, analyze_left_margin

# ---------------------------------------------------------------------------
# Edit this to inspect a different image, then run:  python detect_right_margin.py
# A path given on the command line overrides it.
IMAGE_PATH = r'C:\Users\Shrey\Documents\Margin-Detection\images\Image_79.jpg'
# ---------------------------------------------------------------------------


def analyze_right_margin(image_path, results=None, image=None,
                         visualize=False, verbose=False):
    """Measure right-margin offsets. Same logic as the left, mirrored."""
    if image is None:
        image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at the path: {image_path}")
    if results is None:
        results = get_reader().readtext(image)
    if not results:
        if verbose:
            print("No text detected in the image.")
        return None

    out = analyze_left_margin(image_path, results=results, image=image,
                              visualize=False, verbose=False, mirror=True)
    if out is None:
        return None

    if verbose:
        print(f"Right Margin ({len(out['selected'])} words in band):")
        print("  top:    ", out["filtered_top"])
        print("  mid:    ", out["filtered_mid"])
        print("  bottom: ", out["filtered_bottom"])

    if visualize:
        _draw(image, image_path, out)

    return out


def _draw(image, image_path, out):
    vis = image.copy()
    h, w = vis.shape[:2]
    thick = max(2, int(round(h / 500)))

    y1b, y2b = out["zone_bounds"]
    for yb in (y1b, y2b):
        cv2.line(vis, (0, int(yb)), (w, int(yb)), (160, 160, 160), thick)

    for (ox1, oy1, ox2, oy2) in out["selected"]:
        cv2.rectangle(vis, (int(ox1), int(oy1)), (int(ox2), int(oy2)),
                      (0, 140, 255), thick)

    plt.figure(figsize=(11, 14))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"{os.path.basename(image_path)} - RIGHT margin via mirrored "
              f"left-margin logic\norange = words the band selected  "
              f"({len(out['selected'])})   grey = zone splits")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    print(f"Inspecting {path}")
    res = analyze_right_margin(path, visualize=True, verbose=True)
    if res is None:
        print("Right margin undetermined for this image.")
